import zlib
import base64
import json
from base64 import b64decode
import os

from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig

def extract_orig_elements(orig_elements):
    decoded_orig_elements = base64.b64decode(orig_elements)
    decompressed_orig_elements = zlib.decompress(decoded_orig_elements)
    return decompressed_orig_elements.decode('utf-8')

def parse_unstructured_output(output_file):
    filename = "output_file"
    with open(filename, "rb") as f:
        data = f.read()
    unstructured_json = json.loads(data)

    processed_chunks = []
    for chunk in unstructured_json:
        orig = chunk["metadata"]["orig_elements"]
        orig = extract_orig_elements(orig)
        orig_json = json.loads(orig)
        pages = []
        images = []
        for i, x in enumerate(orig_json):
            metadata = x["metadata"]
            if "image_base64" in metadata:
                im = metadata["image_base64"]

                counter = 0
                start_context = []
                while i - counter > 0:
                    if orig_json[i - counter]["type"] in ["NarrativeText"]:
                        start_context.append(orig_json[i - counter]["text"])
                        if len(start_context) > 5:
                            break
                    counter += 1
                end_context = []
                counter = 1
                while i + counter < len(orig_json):
                    if orig_json[i + counter]["type"] in ["NarrativeText"]:
                        end_context.append(orig_json[i + counter]["text"])
                        if len(end_context) > 3:
                            break
                    counter += 1
                start_context = " ".join(start_context)
                end_context = " ".join(end_context)
                images.append({"base64": im, "start_context": start_context, "end_context": end_context})
            pages.append(metadata["page_number"])

        processed_chunk = {"pages": [min(pages), max(pages)], "text_size": len(chunk["text"]), "text": chunk["text"]}

        if len(images) > 0:
            processed_chunk["images"] = images
        processed_chunks.append(processed_chunk)



def unstructured_partition(input_file, working_directory, api_key=None,
                       partition_endpoint = "https://api.unstructuredapp.io"):
    Pipeline.from_configs(
        context=ProcessorConfig(),
        indexer_config=LocalIndexerConfig(input_path=input_file),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key= api_key if api_key is not None else os.environ.get("UNSTRUCTURED_API_KEY"),
            partition_endpoint=partition_endpoint,
            strategy="hi_res",
            additional_partition_args={
                "coordinates": True,
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 1,
                "extract_image_block_types": ["Image"]
                # "chunking_strategy":"by_title",
                # "combine_text_under_n_chars":300,
                # "max_characters":5000
            }
        ),
        uploader_config=LocalUploaderConfig(output_dir=working_directory)
    ).run()
