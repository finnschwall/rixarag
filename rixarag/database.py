import os
import time

from . import settings
import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Any, Dict, cast

_model = None


class GPUEmbeddings(EmbeddingFunction[Documents]):
    """
    SentenceTransformer with automatic choice of GPU or CPU
    """

    def __init__(self, model, normalize_embeddings: bool = True):
        global _model
        self._normalize_embeddings = normalize_embeddings
        if not _model:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model)

    def __call__(self, input: Documents):
        global _model
        embedding = [_model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )[0]]
        return [i.tolist() for i in embedding]


class RagDB:
    """
    Singleton class to manage the RagDB client and SentenceTransformer embedding function
    """
    _instance = None

    def __init__(self):
        self.value = None
        self.sentence_transformer_ef = GPUEmbeddings(model=settings.EMBEDDING_MODEL, normalize_embeddings=False)
        # embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL, device=device)
        if settings.CHROMA_PERSISTENCE_PATH == "" or settings.CHROMA_PERSISTENCE_PATH is None:
            self.client = chromadb.Client(settings=Settings(allow_reset=True))
        else:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSISTENCE_PATH,
                                                    settings=Settings(allow_reset=True))

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RagDB, cls).__new__(cls, *args, **kwargs)
        return cls._instance


def get_chroma_client():
    """
    Get the ChromaDB client
    """
    return RagDB().client


def purge():
    """
    Reset the ChromaDB. Not reversible!
    """
    client = RagDB().client
    os.environ["ALLOW_RESET"] = "TRUE"
    client.reset()


def delete_collection(collection_name):
    """
    Delete a collection from the ChromaDB. Not reversible!
    """
    client = RagDB().client
    client.delete_collection(name=collection_name)


def add_processed_chunks(processed_chunks, collection_name):
    """
    Add processed chunks to the ChromaDB

    Assumes a processed chunk is a dictionary with the following keys:
    - text: the text of the chunk
    - metadata: a dictionary with metadata about the chunk

    :param processed_chunks: a list of processed chunks
    :param collection_name: the name of the collection to add the chunks to
    """
    ragdb = RagDB()
    collection = ragdb.client.get_or_create_collection(name=collection_name,
                                                       embedding_function=ragdb.sentence_transformer_ef,
                                                       metadata={"hnsw:space": "cosine"})
    for chunk in tqdm(processed_chunks):
        collection.upsert(
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
            ids=[str(hash(chunk["text"]))]
        )


def query(query: str, collection_name="default", n_results: int = 5, kwargs: Dict[str, Any] = {}):
    """
    Query the ChromaDB

    :param query: the query string
    :param collection_name: the name of the collection to query
    :param n_results: the number of results to return
    :param kwargs: additional keyword arguments to pass to the query function of chromadb
    """
    ragdb = RagDB()
    collection = ragdb.client.get_collection(name=collection_name)
    embedding = ragdb.sentence_transformer_ef([query])
    results = collection.query(embedding, n_results=n_results, **kwargs)
    return results


def list_collections():
    """
    List all collections in the db
    """
    ragdb = RagDB()
    return ragdb.client.list_collections()


def get_element_count(collection_name):
    """
    Get the number of elements in a collection
    """
    ragdb = RagDB()
    collection = ragdb.client.get_collection(name=collection_name)
    return collection.count()


def get_all_elements(collection_name):
    """
    Get all elements in a collection
    """
    ragdb = RagDB()
    collection = ragdb.client.get_collection(name=collection_name)
    existing_count = collection.count()
    batch_size = 10
    elems = []
    for i in range(0, existing_count, batch_size):
        batch = collection.get(
            include=["metadatas", "documents", "embeddings"],
            limit=batch_size,
            offset=i)
        elems.extend(batch)
    return elems
