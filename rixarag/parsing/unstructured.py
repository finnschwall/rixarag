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

def parse_unstructured_output(unstructured_json):
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

        processed_chunk = {"text": chunk["text"], "id": hex(abs(hash(chunk["text"])))}

        if min(pages) == max(pages):
            page = str(min(pages))
        else:
            page = f"{min(pages)}-{max(pages)}"
        metadata = {"page": page, "size": len(chunk["text"]), "type": "text"}
        processed_chunk["metadata"] = metadata

        if len(images) > 0:
            processed_chunk["images"] = images
        processed_chunks.append(processed_chunk)
    return processed_chunks



def unstructured_partition(input_file, working_directory, api_key=None,
                       partition_endpoint = "https://api.unstructuredapp.io",
                           max_characters= 5000, combine_text_under_n_chars=300, chunking_strategy="by_title"):
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
                "extract_image_block_types": ["Image"],
                "chunking_strategy":chunking_strategy,
                "combine_text_under_n_chars": combine_text_under_n_chars,
                "max_characters": max_characters
            }
        ),
        uploader_config=LocalUploaderConfig(output_dir=working_directory)
    ).run()




def plot_pdf_with_boxes(pdf_page, segments):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    categories = set()
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    for segment in segments:
        if "coordinates" in segment:
            points = segment["coordinates"]["points"]
            layout_width = segment["coordinates"]["layout_width"]
            layout_height = segment["coordinates"]["layout_height"]
            scaled_points = [
                (x * pix.width / layout_width, y * pix.height / layout_height)
                for x, y in points
            ]
            box_color = category_to_color.get(segment["filetype"], "deepskyblue")
            categories.add(segment["filetype"])
            rect = patches.Polygon(
                scaled_points, linewidth=1, edgecolor=box_color, facecolor="none"
            )
            ax.add_patch(rect)

    # Make legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(
                patches.Patch(color=category_to_color[category], label=category)
            )
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.show()


def render_page(doc_list: list, page_number: int, print_text=True,file_path=None) -> None:
    import fitz
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [
        doc for doc in doc_list if doc["metadata"].get("page_number") == page_number
    ]
    segments = [doc["metadata"] for doc in page_docs]
    plot_pdf_with_boxes(pdf_page, segments)
    if print_text:
        for doc in page_docs:
            print(f"{doc.page_content}\n")