import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import json
import os
from typing import List, Set, Optional
import logging
from pathlib import Path
import time

class WebScraper:
    def __init__(self, base_url: str, download_dir: str,  scrape_recursively=False, delay=0,
                 scrape_only_html=True, save_flat = True, no_scrape=False, additional_metadata=None):
        """
        Initialize the web scraper.

        :param base_url: Base URL to start scraping from
        :param download_dir: Directory to save downloaded files
        :param scrape_recursively: Whether to scrape pages recursively (i.e. look on all downloaded pagesfor more links)
            or just the base page
        :param delay: Delay between requests in seconds
        :param scrape_only_html: Whether to only scrape HTML pages
        :param save_flat: Whether to save all files in the same directory (create no subdirectories). Can cause name conflicts
        :param no_scrape: Whether to skip scraping and only download the base page
        :param additional_metadata: Additional metadata to save to each file .metadata.json
        """
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.download_dir = download_dir
        self.processed_urls: Set[str] = set()
        self.scrape_recursively = scrape_recursively
        self._scraped_links = False
        self.delay = delay
        self.scrape_only_html = scrape_only_html
        self.save_flat = save_flat
        self.no_scrape = no_scrape


        # Create download directory if it doesn't exist
        Path(download_dir).mkdir(parents=True, exist_ok=True)

    def clean_url(self, url: str) -> Optional[str]:
        """
        Clean and validate a URL.
        :param url: URL to clean

        :return: Cleaned URL or None if invalid
        """
        # Remove fragment identifier
        url, _ = urldefrag(url)

        # Skip empty URLs or fragment-only URLs
        if not url or url.startswith('#'):
            return None

        # Make relative URLs absolute
        if not bool(urlparse(url).netloc):
            url = urljoin(self.base_url, url)

        return url

    def is_valid_url(self, url: str, internal_only: bool = True, blacklist: Optional[List[str]] = None) -> bool:
        """
        Check if URL should be processed based on rules.

        :param url: URL to check
        :param internal_only: Whether to only allow internal URLs
        :param blacklist: List of domains to exclude

        Returns:
            Boolean indicating if URL should be processed
        """
        parsed_url = urlparse(url)

        # Skip invalid URLs
        if not parsed_url.scheme or not parsed_url.netloc:
            return False

        # Check if internal only
        if internal_only and parsed_url.netloc != self.base_domain:
            return False

        # Check blacklist
        if blacklist:
            if any(blocked in url for blocked in blacklist):
                return False

        return True

    def save_file(self, url: str, content: bytes, metadata: dict) -> None:
        """
        Save downloaded file and its metadata.

        :param url: URL of the file
        :param content: File content
        :param metadata: Metadata to save
        """
        # Create filename from URL
        parsed_url = urlparse(url)

        if self.save_flat:
            filepath = os.path.basename(parsed_url.path.strip('/'))
            full_path = os.path.join(self.download_dir, filepath)
        else:
            filepath = parsed_url.path.strip('/')
            if not filepath:
                filepath = 'index.html'
            full_path = os.path.join(self.download_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if len(filepath.split(".")) < 2:
            if "html" in metadata['content_type']:
                full_path += ".html"
            elif "json" in metadata['content_type']:
                full_path += ".json"


        # Save file
        with open(full_path, 'wb') as f:
            f.write(content)

        # Save metadata
        metadata_path = f"{full_path}.metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def scrape(self, internal_only: bool = True, blacklist: Optional[List[str]] = None, recursive=False) -> None:
        """
        Start the scraping process.


        :param internal_only: Whether to only process internal URLs (i.e. ignore references to websites of other domains)
        :param blacklist: List of domains to exclude (only used if internal_only is False)
        """
        urls_to_process = {self.base_url}

        while urls_to_process:
            current_url = urls_to_process.pop()

            if current_url in self.processed_urls:
                continue

            self.processed_urls.add(current_url)

            try:
                # Download content
                time.sleep(self.delay)
                response = requests.get(current_url, timeout=30)
                response.raise_for_status()

                # Save file and metadata
                content_type = response.headers.get('content-type')
                metadata = {
                    'url': current_url,
                    'content_type': content_type,
                    'size': len(response.content),
                    'timestamp': response.headers.get('last-modified')
                }

                if 'text/html' in content_type:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = soup.find('title')
                    title_text = title.string if title else None
                    if title_text:
                        #seems like titles can have junk in them
                        title_text = title_text.replace('\n', '').strip()
                        metadata['title'] = title_text


                if not self.scrape_only_html:
                    self.save_file(current_url, response.content, metadata)
                else:
                    if 'text/html' in content_type:
                        self.save_file(current_url, response.content, metadata)
                    else:
                        print(f"Skipping {current_url} as it is not HTML")

                if self.no_scrape:
                    continue
                if not self.scrape_recursively:
                    if self._scraped_links:
                        continue
                    else:
                        self._scraped_links = True
                if 'text/html' in response.headers.get('content-type', ''):
                    soup = BeautifulSoup(response.content, 'html.parser')
                    all_links = soup.find_all(name="a")

                    for link in all_links:
                        href = link.get('href')
                        if href:
                            cleaned_url = self.clean_url(href)
                            if cleaned_url and self.is_valid_url(cleaned_url, internal_only, blacklist):
                                urls_to_process.add(cleaned_url)

            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")
                continue
        # finish with saving metadata on the entire run
        run_metadata = {"origin": self.base_url,"download_total": len(self.processed_urls),
                        "processed_urls": list(self.processed_urls),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), }
        with open(os.path.join(self.download_dir, "run_metadata.json"), 'w') as f:
            json.dump(run_metadata, f, indent=2)

