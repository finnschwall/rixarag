import json
import re
import warnings

from .parsing import regex_parser
import os, glob
from tqdm import tqdm
import numpy as np
import rixarag.database as database
import time
from markdownify import markdownify as md
import pymupdf

def latex_pipeline(path, document_title=None, collection="default", caption_images=False, working_directory=None,
                   original_pdf=None,
                   desired_chunk_size=1000, hard_maximum_chunk_size=2000, fallback_strategy="split",
                   return_chunks=False):
    """
    Process a latex document (or documents) into chunks and store them in the vector database

    :param path: Path to the latex document or folder containing latex documents
    :param document_title: Title of the document. If not provided, will attempt to extract from the document
    :param collection: Name of the collection in the database the chunks will be stored in
    :param caption_images: Whether to caption images in the document. Experimental
    :param working_directory: Directory to save processed chunks to. Smart if tex is large to avoid reprocessing
    :param original_pdf: Path to the original PDF. If provided, will attempt to match chunks to pages in the PDF
    :param desired_chunk_size: Desired size of chunks in characters
    :param hard_maximum_chunk_size: Hard upper limit for chunk size
    :param fallback_strategy: Strategy to use if chunking fails. See regex_parser for options
    :param return_chunks: Whether to return the processed chunks as a list instead of storing in the database
    :return: None or list of processed chunks
    """
    start_time = time.time()
    if not original_pdf and not document_title:
        raise ValueError("Either original_pdf or document_title must be provided to get title of document!")
    if original_pdf and not document_title:
        # TODO need to extract metadata from pdf
        doc = pymupdf.open(original_pdf)
        document_title = doc.metadata["title"]
        doc.close()
        if not document_title or document_title == "":
            raise ValueError("Could not extract title from PDF. Please provide document_title manually.")

    # check if path is .tex or folder
    tex_files = []

    if os.path.isdir(path):
        for file in glob.glob(os.path.join(path, "*.tex")):
            tex_files.append(file)
    elif os.path.isfile(path):
        tex_files.append(path)
    if not original_pdf:
        print(f"Detected {len(tex_files)} .tex files. Starting chunking...")
    else:
        print(f"Detected {len(tex_files)} .tex files. Starting chunking and matching to original PDF...")
    processed_chunks = []
    if original_pdf:
        from .parsing import pdf_page_matcher
        pdf_page_matcher.reset_total()
    for tex_file in tqdm(tex_files):
        with open(tex_file, "r") as f:
            tex = f.read()

        chunks_raw, metadata = regex_parser.automatic_chunking(tex, desired_chunk_size=desired_chunk_size,
                                                               hard_maximum_chunk_size=hard_maximum_chunk_size,
                                                               fallback_strategy=fallback_strategy)
        processed_chunks.extend(regex_parser.chunks_to_db_chunks_latex(chunks_raw, document_title, original_pdf=original_pdf))
    sizes = [chunk["metadata"]["size"] for chunk in processed_chunks]
    print(
        f"Processed files into a total of {len(processed_chunks)} chunks. Mean size: {round(np.mean(sizes))}, Median: {round(np.median(sizes))}, "
        f"Max: {np.max(sizes)}, Min: {np.min(sizes)}")
    if original_pdf:
        from .parsing import pdf_page_matcher
        print(f"Matched uniquely {pdf_page_matcher.total_matched} out of {pdf_page_matcher.total_total} chunks.")
        total_errors = 0
        for i in processed_chunks:
            if i["metadata"]["page_number"] == "Error":
                total_errors += 1
        if total_errors > 0:
            print(f"Failed to match {total_errors} chunks entirely.")

    if working_directory:
        if not os.path.exists(working_directory):
            warnings.warn("Working directory does not exist. Will proceed without saving...")
        else:
            print(f"Saving processed chunks to {working_directory}")
            with open(os.path.join(working_directory, f"processed_chunks_{document_title}.json"), "w") as f:
                json.dump(processed_chunks, f)
    if return_chunks:
        return processed_chunks
    print("Will now calulate embeddings and transfer into database. This may take a while...")
    database.add_processed_chunks(processed_chunks, collection)
    end_time = time.time()
    print(f"Finished processing in {round(end_time - start_time, 2)} seconds.")


def html_pipeline(path, collection="default", base_link=None, desired_chunk_size=1000, hard_maximum_chunk_size=2000,
                  fallback_strategy="split",
                  return_chunks=False):
    """
    Process a html document (or documents) into chunks and store them in the vector database

    :param path: Path to the html document or folder containing html documents
    :param collection: Name of the collection in the database the chunks will be stored in
    :param base_link: URL for single HTML file. For multiples this will be base_link/filename for each file
    :param desired_chunk_size: Desired size of chunks in characters
    :param hard_maximum_chunk_size: Hard upper limit for chunk size
    :param fallback_strategy: Strategy to use if chunking fails. See regex_parser for options
    :param return_chunks: Whether to return the processed chunks as a list instead of storing in the database
    :return: None or list of processed chunks
    """

    start_time = time.time()
    html_files = []
    if os.path.isdir(path):
        for file in glob.glob(os.path.join(path, "*.html")):
            html_files.append(file)
    elif os.path.isfile(path):
        html_files.append(path)
    if len(html_files) == 0:
        raise ValueError("No .html files found")
    print(f"Detected {len(html_files)} .html files. Starting chunking...")
    processed_chunks = []
    for html_file in html_files:
        with open(html_file, "r") as f:
            html = f.read()
        markdown = md(html)
        page_title = "Unknown"
        parts = markdown.split('\n')
        for line in parts:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('#'):
                page_title = line
                break
        chunks_raw, metadata = regex_parser.automatic_chunking(markdown, desired_chunk_size=desired_chunk_size,
                                                               hard_maximum_chunk_size=hard_maximum_chunk_size,
                                                               fallback_strategy=fallback_strategy,
                                                               document_type="markdown")
        url = None
        if base_link:
            if not base_link.endswith("/"):
                base_link = base_link + "/"
            url = base_link + os.path.basename(html_file)
        processed_chunks.extend(regex_parser.chunks_to_db_chunks_html(chunks_raw, page_title, url))
    sizes = [chunk["metadata"]["size"] for chunk in processed_chunks]
    print(
        f"Processed files into a total of {len(processed_chunks)} chunks. Mean size: {round(np.mean(sizes))}, Median: {round(np.median(sizes))}, "
        f"Max: {np.max(sizes)}, Min: {np.min(sizes)}")
    if return_chunks:
        return processed_chunks
    print("Will now calulate embeddings and transfer into database. This may take a while...")
    database.add_processed_chunks(processed_chunks, collection)
    end_time = time.time()
    print(f"Finished processing in {round(end_time - start_time, 2)} seconds.")
