import codecs
import hashlib
import json
import re
import warnings

from .parsing import regex_parser, unstructured
import os, glob
from tqdm import tqdm
import numpy as np
import rixarag.database as database
import time
from markdownify import markdownify as md
import pymupdf
from typing import Union, List


def latex_pipeline(path, document_title=None, collection="default", caption_images=False, working_directory=None,
                   original_pdf=None,
                   desired_chunk_size=1000, hard_maximum_chunk_size=2000, fallback_strategy="split",
                   return_chunks=False, additional_metadata=None):
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
    :param additional_metadata: Additional metadata to add to the chunks (as dict). Careful as this will overwrite existing keys
    :return: None or list of processed chunks
    """
    start_time = time.time()
    if not original_pdf:
        # the original pdf will be searched for using the exact same name as the tex file
        if not os.path.isdir(path) and path.endswith(".tex"):
            probable_pdf = path.replace(".tex", ".pdf")
            if os.path.exists(probable_pdf):
                original_pdf = probable_pdf
                print(f"Found matching PDF: {original_pdf}")
        else:
            for file in glob.glob(os.path.join(path, "*.tex")):
                probable_pdf = file.replace(".tex", ".pdf")
                if os.path.exists(probable_pdf):
                    original_pdf = probable_pdf
                    print(f"Found matching PDF: {original_pdf}")
    if path:
        metadata_file = None
        if not os.path.isdir(path):
            if os.path.exists(path + ".metadata.json"):
                with open(path + ".metadata.json", "r") as f:
                    metadata_file = json.load(f)
        else:
            for file in glob.glob(os.path.join(path, "*.metadata.json")):
                with open(file, "r") as f:
                    metadata_file = json.load(f)
                break
        if metadata_file:
            if not document_title and "document_title" in metadata_file:
                document_title = metadata_file["document_title"]
            if not document_title and "title" in metadata_file:
                document_title = metadata_file["title"]
            if not additional_metadata:
                additional_metadata = metadata_file
            else:
                additional_metadata.update(metadata_file)

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
    hash_base = ",".join(tex_files) + processed_chunks[0]["text"]
    document_id = regex_parser.generate_id(hash_base)
    for chunk in processed_chunks:
        chunk["metadata"]["document_id"] = document_id
        if additional_metadata:
            chunk["metadata"].update(additional_metadata)
    sizes = [chunk["metadata"]["size"] for chunk in processed_chunks]
    print(
        f"Processed files into a total of {len(processed_chunks)} chunks. Mean size: {round(np.mean(sizes))}, Median: {round(np.median(sizes))}, "
        f"Max: {np.max(sizes)}, Min: {np.min(sizes)}")
    if original_pdf:
        from .parsing import pdf_page_matcher
        print(f"Matched uniquely {pdf_page_matcher.total_matched} out of {pdf_page_matcher.total_total} chunks.")
        total_errors = 0
        for i in processed_chunks:
            if i["metadata"]["page"] == "Error":
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


def markdown_pipeline(path=None, markdown_texts: List = None, collection="default",
                      desired_chunk_size=1000, hard_maximum_chunk_size=2000, fallback_strategy="split",
                      return_chunks=False, title: Union[str, List[str]] = None,
                      additional_metadata : Union[dict, List[dict]]=None):
    """
    Process a markdown document (or documents) into chunks and store them in the vector database

    :param path: Path to the html document or folder containing html documents. Will automatically retrieve text from files
    :param markdown_texts: List of markdown texts (or just a text) to process
    :param collection: Name of the collection in the database the chunks will be stored in
    :param desired_chunk_size: Desired size of chunks in characters
    :param hard_maximum_chunk_size: Hard upper limit for chunk size
    :param fallback_strategy: Strategy to use if chunking fails. See regex_parser for options
    :param return_chunks: Whether to return the processed chunks as a list instead of storing in the database
    :param title: Title of the document. Either one title for all passed elements or a list for each.
        If not provided, will attempt to extract.
    :param additional_metadata: Additional metadata to add to the chunks (as dict). Either provide one for all or a list for each document.
        Careful as this will overwrite existing keys
    :return: None or list of processed chunks
    """

    start_time = time.time()
    markdown_files = []
    if path:
        markdown_texts=[]
        if os.path.isdir(path):
            for file in glob.glob(os.path.join(path, "*.md")):
                markdown_files.append(file)
        elif os.path.isfile(path):
            markdown_files.append(path)
        if len(markdown_files) == 0:
            raise ValueError("No .md files found")
        print(f"Detected {len(markdown_files)} .md files. Starting chunking...")
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                markdown = f.read()
            markdown_texts.append(markdown)
    elif markdown_texts:
        if markdown_texts is str:
            markdown_texts = [markdown_texts]
        else:
            markdown_files = markdown_texts
    else:
        raise ValueError("No path or texts provided")
    processed_chunks = []
    for i, markdown in enumerate(markdown_texts):
        if not title:
            cur_title = regex_parser.extract_heading(markdown)
        else:
            if isinstance(title, str):
                cur_title = title
            else:
                cur_title = title[i]
        markdown = regex_parser.clean_md(markdown)

        chunks_raw, metadata = regex_parser.automatic_chunking(markdown, desired_chunk_size=desired_chunk_size,
                                                               hard_maximum_chunk_size=hard_maximum_chunk_size,
                                                               fallback_strategy=fallback_strategy,
                                                               document_type="markdown")

        processed_chunks.extend(regex_parser.chunks_to_db_chunks_html(chunks_raw, cur_title))
        if additional_metadata:
            for chunk in processed_chunks:
                if isinstance(additional_metadata, dict):
                    add_metadata = additional_metadata
                else:
                    add_metadata = additional_metadata[i]
                filtered_additional_metadata = {
                    key: value
                    for key, value in add_metadata.items()
                    if key not in ["size", "id"]
                }
                chunk["metadata"].update(filtered_additional_metadata)

    hash_base = ",".join(markdown_files) + processed_chunks[0]["text"]
    # just joining filenames may risk collision when ingesting files with same name
    document_id = regex_parser.generate_id(hash_base)
    for chunk in processed_chunks:
        chunk["metadata"]["document_id"] = document_id
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

def html_pipeline(path=None, html_texts: List = None, collection="default", base_link=None,
                        desired_chunk_size=1000, hard_maximum_chunk_size=2000, fallback_strategy="split",
                        return_chunks=False, title: Union[str, List[str]] = None,
                  additional_metadata: Union[dict, List[dict]]=None):
    """
    Process a html file (or file) into chunks and store them in the vector database

    :param path: Path to the html document or folder containing html documents. Will automatically retrieve text from files
    :param markdown_texts: List of markdown texts (or just a text) to process
    :param collection: Name of the collection in the database the chunks will be stored in
    :param base_link: URL for single HTML file. This will be used to generate URLs like base_link + filename
    :param desired_chunk_size: Desired size of chunks in characters
    :param hard_maximum_chunk_size: Hard upper limit for chunk size
    :param fallback_strategy: Strategy to use if chunking fails. See regex_parser for options
    :param return_chunks: Whether to return the processed chunks as a list instead of storing in the database
    :param title: Title of the document. Either one title for all passed elements or a list for each.
        If not provided, will attempt to extract.
    :param additional_metadata: Additional metadata to add to the chunks (as dict). Either provide one for all or a list for each document.
        Careful as this will overwrite existing keys
    :return: None or list of processed chunks
    """
    html_contents = []
    if path:
        html_files = []
        urls = []
        meta = []
        if os.path.isdir(path):
            for file in glob.glob(os.path.join(path, "*.html")):
                html_files.append(file)
        elif os.path.isfile(path):
            html_files.append(path)
        if base_link:
            if not base_link.endswith("/"):
                base_link = base_link + "/"
            for html_file in html_files:
                url = base_link + os.path.basename(html_file)
                urls.append(url)
        for html_file in html_files:
            with codecs.open(html_file, 'r', encoding='utf-8',
                             errors='ignore') as fdata:
                # html files are often downloaded and can have all sorts of encodings.
                # this is not perfect as things get lost but its better than just giving up
                html = fdata.read()
            # with open(html_file, "r") as f:
            #     html = f.read()
            html_contents.append(md(html))
        for i, html_file in enumerate(html_files):
            meta_single = {"source_file": os.path.basename(html_file)}
            if len(urls) >0:
                meta_single["url"] = urls[i]
            if additional_metadata:
                if isinstance(additional_metadata, dict):
                    meta_single.update(additional_metadata)
                else:
                    meta_single.update(additional_metadata[i])
            meta.append(meta_single)

            if os.path.exists(html_file+".metadata.json"):
                with open(html_file+".metadata.json", "r") as f:
                    meta_single.update(json.load(f))
    elif html_texts:
        if isinstance(html_texts, str):
            html_texts = [html_texts]
        html_contents = html_texts
    else:
        raise ValueError("No path or html text(s) provided")
    return markdown_pipeline(markdown_texts=html_contents, collection=collection, desired_chunk_size=desired_chunk_size,
                      hard_maximum_chunk_size=hard_maximum_chunk_size, fallback_strategy=fallback_strategy,
                      return_chunks=return_chunks, title=title, additional_metadata=meta)



def _read_dir(data_dir, collection):
    """
    Read a directory, try to find a fitting pipeline and then process the files in it.
    Subdirs are now ignored
    """
    assumed_pipeline = None
    if os.path.basename(data_dir) in ["images"]:
        return
    files_found = False
    for file in os.listdir(data_dir):
        if not files_found:
            if os.path.isfile(file):
                files_found = True
        if file.endswith(".tex"):
            assumed_pipeline = "tex"
            break
        elif file.endswith(".md"):
            assumed_pipeline = "md"
            break
        elif file.endswith(".html"):
            assumed_pipeline = "html"
            break
    if not assumed_pipeline and files_found:
        print(f"No fitting pipeline found for this directory.")
    print(f"\n\nProcessing {data_dir}")
    if assumed_pipeline == "tex":
        # any pdf in the folder is assumed to be the original pdf
        original_pdf = None
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                original_pdf = os.path.join(data_dir, file)
                break
        latex_pipeline(data_dir, collection=collection, original_pdf=original_pdf)
    elif assumed_pipeline == "md":
        markdown_pipeline(data_dir, collection=collection)
    elif assumed_pipeline == "html":
        html_pipeline(data_dir, collection=collection)
    print("-" * 10)



def read_directories(path, collection="default"):
    """
    Read a directory and all its subdirectories and store them the contents into the database.

    Each subdirectory is considered and treated separately. However a directory is considered a single document.
    The correct pipeline for each directory is based on the present file types.
    :param path:
    :return:
    """
    failed_paths = []
    count=0
    for root, dirs, files in os.walk(path):
        try:
            _read_dir(root, collection)
            count+=1
        except Exception as e:
            raise e
            print(f"Error processing {root}. Skipping...")
            print(e)
            failed_paths.append(root)
    print(f"Finished! Processed {count} directories.")
    if len(failed_paths) > 0:
        print("------\nFailed to process the following directories:")
        for path in failed_paths:
            print(path)




def unstructured_loader_pipeline(unstructured_path, embed_images=False, collection="default",
                                 return_chunks=False, document_title = None, source_file = None,
                                 working_directory=None):
    """
    Load a json output from the unstructured API and process it into the database

    :param unstructured_path:
    :return:
    """
    with open(unstructured_path, "rb") as f:
        data = f.read()
    unstructured_json = json.loads(data)
    document_id = regex_parser.generate_id(unstructured_path)
    processed_chunks = unstructured.parse_unstructured_output(unstructured_json)
    if not source_file:
        source_file = os.path.basename(unstructured_path)
    if not document_title:
        print("No document title provided. Will use filename.")
        document_title = os.path.basename(unstructured_path).split(".json")[0]
    for chunk in processed_chunks:
        chunk["metadata"]["source_file"] = source_file
        chunk["metadata"]["document_title"] = document_title
        chunk["metadata"]["document_id"] = str(document_id)
        chunk["metadata"]["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


    if not embed_images:
        for chunk in processed_chunks:
            if "images" in chunk:
                del chunk["images"]
    else:
        from .parsing import image_captioning
        processed_chunks, count, total_token_count = image_captioning.caption_images(processed_chunks)
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
