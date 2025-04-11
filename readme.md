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
- Docx, ODT and other formats -> chunks (requires pandoc)
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
### Markdown
Turn a whole folder of markdown files (or just one file) from the internet into embeddings:

```python
from rixarag import pipelines

pipelines.markdown_pipeline("PATH_TO_FOLDER")
```
### HTML
Turn a whole folder of HTML files (or just one file) from the internet into embeddings:
```python
from rixarag import pipelines
pipelines.html_pipeline("PATH_TO_FOLDER")
```

Look at the docs for how to add the urls or other features.

### HTML from a website
Turn a website into embeddings:
```python
from rixarag.tools import webscraper
scraper = webscraper.Webscraper("URL_TO_SCRAPE", "PATH_TO_STORE_HTML", delay=0.1)
scraper.scrape()
from rixarag import pipelines
pipelines.html_pipeline("PATH_TO_STORE_HTML")
```
See documentation for more control (e.g. scraping really everything)

### Latex
Turn a whole folder of Latex files (e.g. a book source) into embeddings:
```python
from rixarag import pipelines
chunks = pipelines.latex_pipeline("PATH_TO_FOLDER",document_title="How the books called", 
        original_pdf = "PATH_TO_COMPILED_PDF")
```
You dont need to provide the original PDF. However if you do the chunker will attempt to give page numbers to the chunks.


### Wikipedia
Coming

### Unstructured
Coming

### Making things permanent
```python
import rixarag
rixarag.settings.CHROMA_PERSISTENCE_PATH = "path/to/where/you/want/to/store/chroma"
```
### Using GPU
Will do automatically, if pytorch is installed with CUDA support.

### Just want to use this for my own database
Set the `return_chunks` parameter to `True` in the pipeline functions.

Otherwise, to get .json set the `working_directory` parameter to a folder. But this just
saves the chunks as .json files.

## Documentation
For a full documentation, see the [documentation](https://finnschwall.github.io/rixarag/).