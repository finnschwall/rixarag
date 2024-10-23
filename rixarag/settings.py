IMAGE_CAPTIONING_BACKEND = "openai"
""" The backend to use for image captioning. Choose between "openai" and "llamacpp"
"""
OPENAI_IMAGE_MODEL = "gpt-4o-mini"
""" The model to use for image captioning. Only relevant if IMAGE_CAPTIONING_BACKEND is set to "openai"
"""
LLAMACPP_IMAGE_MODEL = "default"
""" Path to the .gguf file to use for image captioning. Only relevant if IMAGE_CAPTIONING_BACKEND is set to "llamacpp"
"""

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
""" The model used for calculating embeddings.
See [this](https://huggingface.co/spaces/mteb/leaderboard) leaderboard for more models.
Especially for non-enlgish languages, you might want to choose a different model.
"""

USE_GPU = False

CHROMA_PERSISTENCE_PATH = ""
""" The path to the chroma database. If empty, the database will be stored in memory.
But then it will be lost when the program is closed."""

CUSTOM_CHROMA_INIT = None
"""Set a function pointer to a custom initialization function for the chroma database.
Can be used to connect to a remote
"""

DELETE_THRESHOLD = 30
"""Chunks/texts under this limit will not be stored in the database.
"""

IMAGE_CAPTIONING_INSTRUCTION_TEMPLATE = """You are an assistant tasked with summarizing images for RAG retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the image that is well optimized for retrieval.
Keep in mind that the image descriptions will be embedded "on their own" i.e. not as part of the body of text around it. Be sufficiently descriptive!

The images are retrieved from a document via OCR. Some elements should not be embedded e.g. logos. In such a case answer only with "SKIP".

To properly caption the image you are given context below from where the image was retrieved. It is possible that the "fitting" context may not be included yet, due to the original documents structure.
Hence do not "SKIP" out images that do not fit the context, but look like they are part of some other context. Try to give a description anyway.



CONTEXT:
-----
{start_context}
-----
IMAGE
-----
{end_context}"""