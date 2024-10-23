## Introduction

RAG system for highly structured documents.

Developed for
[RIXA](https://github.com/finnschwall/rixa).
but works standalone.

## Main features
- Chunk Latex into blocks fit for a RAG system based on the author's used sequences (\section, \subsection...)
- Chunk -> PDF pages for Latex chunks. Allows for easy finding of source material.
- HTML -> chunks
- Markdown -> chunks
- Wikipedia XML -> chunks
- And of course: A database where to store and retrieve the chunks
- Multimodal support: Automatically transcribe and embedd images
- Use any document with the "unstructured" API (bring your own key)
- Various comfort features (Add URLs to chunks from HTML, Wikipedia section URLS...)

## Quickstart
Install with
```bash
pip3 install git+https://github.com/finnschwall/rixarag.git
```
(PIP coming)

## Example
### HTML or Markdown
Turn a whole folder of HTML scraped from the internet into embeddings:
```python
from rixarag import pipelines
pipelines.html_pipeline("PATH_TO_FOLDER", base_link="https://www.where_you_scraped_from.com")
```
And that's it! You can now query the database for the websites contents.
Usage for single file is the same. For Markdown choose the markdown_pipeline.
### Latex
Turn a whole folder of Latex files (e.g. a book source) into embeddings:
```python
from rixarag import pipelines
chunks = pipelines.latex_pipeline("PATH_TO_FOLDER",document_title="How the books called", 
        original_pdf = "PATH_TO_COMPILED_PDF")
```
If you don't provide the original PDF, everything still works, but you can't get the page numbers.
Usage for single file is the same.

### Wikipedia
Coming

### Unstructured
Coming

### Making things permanent
```python
import rixarag
rixarag.settings.CHROMA_PERSISTENCE_PATH = "path/to/where/you/want/to/store/chroma"
```

### Just want to use` this for my own database
Set the `return_chunks` parameter to `True` in the pipeline functions.

Otherwise, to get .json set the `working_directory` parameter to a folder. But this just
saves the chunks as .json files.

## Documentation
For a full documentation, see the [documentation](https://finnschwall.github.io/rixarag/).