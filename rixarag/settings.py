from decouple import Config, RepositoryEnv, Csv, Choices, AutoConfig
import os

config_dir = "."
try:
    config_dir = os.environ["RIXA_WD"]
    config = Config(RepositoryEnv(os.path.join(config_dir, "config.ini")))
except KeyError:
    current_directory = os.getcwd()
    files = os.listdir(current_directory)
    if "config.ini" in files:
        config_dir = current_directory
        config = Config(RepositoryEnv(os.path.join(config_dir, "config.ini")))
    else:
        config_dir = os.path.abspath(config_dir)
        config = AutoConfig()

IMAGE_CAPTIONING_BACKEND = config("IMAGE_CAPTIONING_BACKEND", default="openai", cast=str)
""" The backend to use for image captioning. Choose between "openai" and "llamacpp"
"""
OPENAI_IMAGE_MODEL = config("OPENAI_IMAGE_MODEL", default="gpt-4o-mini", cast=str)
""" The model to use for image captioning. Only relevant if IMAGE_CAPTIONING_BACKEND is set to "openai"
"""
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)
LLAMACPP_IMAGE_MODEL = config("LLAMACPP_IMAGE_MODEL", default="default", cast=str)
""" Path to the .gguf file to use for image captioning. Only relevant if IMAGE_CAPTIONING_BACKEND is set to "llamacpp"
"""

#mixedbread-ai/mxbai-embed-large-v1
EMBEDDING_MODEL = config("EMBEDDING_MODEL", default="sentence-transformers/all-MiniLM-L6-v2", cast=str)
""" The model used for calculating embeddings.
See [this](https://huggingface.co/spaces/mteb/leaderboard) leaderboard for more models.
Especially for non-enlgish languages, you might want to choose a different model.
"""

USE_CROSS_ENCODER = config("USE_CROSS_ENCODER", default=True, cast=bool)
CROSS_ENCODER_MODEL = config("CROSS_ENCODER_MODEL", default="mixedbread-ai/mxbai-rerank-xsmall-v1", cast=str)

FORCE_DEVICE_CROSS_ENCODER = config("FORCE_DEVICE", default=None)
"""Will manually set the device= of the sentence transformer model.
If none sentence transformer will choose the device automatically.
"""

FORCE_DEVICE_EMBEDDING_MODEL = config("FORCE_DEVICE_EMBEDDING_MODEL", default=None)

CHROMA_PERSISTENCE_PATH = config("CHROMA_PERSISTENCE_PATH", default="", cast=str)
""" The path to the chroma database. If empty, the database will be stored in memory.
But then it will be lost when the program is closed."""

CUSTOM_CHROMA_INIT = config("CUSTOM_CHROMA_INIT", default=None)
"""Set a function pointer to a custom initialization function for the chroma database.
Can be used to connect to a remote
"""

DELETE_THRESHOLD = config("DELETE_THRESHOLD", default=100, cast=int)
"""Chunks/texts under this limit will not be stored in the database.
"""

IMAGE_CAPTIONING_INSTRUCTION_TEMPLATE = config("IMAGE_CAPTIONING_INSTRUCTION_TEMPLATE", default="""You are an assistant tasked with summarizing images for RAG retrieval.
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
{end_context}""", cast=str)