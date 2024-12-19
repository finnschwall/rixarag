import json
import os
import random
import time

from . import settings
import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Any, Dict, cast
import numpy as np
_model = None
from sentence_transformers import CrossEncoder


import logging

faulty_logger = logging.getLogger("chromadb.telemetry.product.posthog")
faulty_logger.setLevel(logging.CRITICAL)


class GPUEmbeddings(EmbeddingFunction[Documents]):
    """
    SentenceTransformer with automatic choice of GPU or CPU
    """

    def __init__(self, model, normalize_embeddings: bool = True):
        global _model
        self._normalize_embeddings = normalize_embeddings
        if not _model:
            from sentence_transformers import SentenceTransformer
            if settings.FORCE_DEVICE:
                _model = SentenceTransformer(model, device=settings.FORCE_DEVICE)
            else:
                _model = SentenceTransformer(model)

    def encode(self, input ):
        return _model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )

    def __call__(self, input: Documents):
        global _model
        embedding = [_model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )[0]]
        return [i.tolist() for i in embedding]


def load_model():
    """
    Just load the model. Useful to reserve GPU memory
    :return:
    """
    GPUEmbeddings(model=settings.EMBEDDING_MODEL, normalize_embeddings=False)

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
            self.client = chromadb.Client(settings=Settings(allow_reset=True,anonymized_telemetry=False))
        else:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSISTENCE_PATH,
                                                    settings=Settings(allow_reset=True,anonymized_telemetry=False))
        if settings.USE_CROSS_ENCODER:
            self.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL, device=settings.FORCE_DEVICE)

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
    unique_documents = {}
    used_metadata_keys = set()
    for chunk in tqdm(processed_chunks):
        # could be done in one go but a progress bar is nice

        # fix None values sneaking in and causing problems
        metadata = chunk["metadata"]
        metadata = {
            key: value
            for key, value in metadata.items()
            if value
        }
        problematic_characters = ["\n", "#", "¶", "[", "]"]
        # everything related to markdown is problematic as it may interfere with the front end
        for key, value in metadata.items():
            if isinstance(value, str):
                new_value = value
                for char in problematic_characters:
                    new_value = new_value.replace(char, "")
                metadata[key] = new_value
        metadata["id"] = chunk["id"]
        # used_metadata_keys.update(metadata.keys())
        if "document_id" in metadata:
            if not metadata["document_id"] in unique_documents:
                doc_meta = metadata
                doc_id = metadata["document_id"]
                relevant = ["source_type", "source_file", "source", "document_title", "authors", "url", "content_type",
                            "publisher", "timestamp", "type", "date", "date_published","info", "creation_date"]
                doc_dic = {}
                for key in relevant:
                    if key in doc_meta:
                        doc_dic[key] = doc_meta[key]
                unique_documents[doc_id] = doc_meta



        if "images" in chunk:
            for image in chunk["images"]:
                metadata["size"] = len(image["transcription"])
                metadata["type"] = "image"
                metadata["base64"] = image["base64"]
                metadata["chunk_id"] = chunk["id"]
                collection.upsert(
                    documents=[image["transcription"]],
                    metadatas=[metadata],
                    ids=[hex(abs(hash(image["transcription"])))[2:]]
                )
        if settings.DELETE_THRESHOLD:
            if len(chunk["text"]) < settings.DELETE_THRESHOLD:
                continue
        collection.upsert(
            documents=[chunk["text"]],
            metadatas=[metadata],
            ids=[chunk["id"]]
        )
    if settings.CHROMA_PERSISTENCE_PATH:
        # metapath = settings.CHROMA_PERSISTENCE_PATH + "/metadata.txt"
        # with open(metapath, "a") as f:
        #     f.write(str(used_metadata_keys)+"\n")
        document_path = settings.CHROMA_PERSISTENCE_PATH + "/documents.json"
        try:
            if os.path.exists(document_path):
                with open(document_path, "r+") as f:
                    previous = json.loads(f.read())
                    previous.update(unique_documents)
                    f.seek(0)
                    f.write(json.dumps(previous))
            else:
                with open(document_path, "w") as f:
                    f.write(json.dumps(unique_documents))
        except Exception as e:
            print(f"Encountered error while writing document metadata. This does not affect the database. {e}")


def query(query_str: str, collection="default", n_results: int = 5, max_distance = 0.75, kwargs: Dict[str, Any] = {},):
    """
    Query the ChromaDB

    :param query: the query string
    :param collection: the name of the collection to query
    :param n_results: the number of results to return
    :param max_distance: Entries with distances larger than this will be filtered out
    :param kwargs: additional keyword arguments to pass to the query function of chromadb
    """
    # TODO what effect does the embedding function have on max_distance?
    ragdb = RagDB()
    collection = ragdb.client.get_collection(name=collection)
    embedding = ragdb.sentence_transformer_ef([query_str])
    results = collection.query(embedding, n_results=n_results, **kwargs)
    # this results in a key : [[entries]] structure as we only ever do one query but query takes usually multiples.
    for key, value in results.items():
        if value:
            results[key] = value[0]
    valid_indices = [i for i, dist in enumerate(results['distances']) if dist <= max_distance]
    filtered_data = {}
    for key in results:
        if key == 'embeddings' and results[key] is None:
            filtered_data[key] = None
        elif isinstance(results[key], list):
            filtered_data[key] = [results[key][i] for i in valid_indices]
        else:
            filtered_data[key] = results[key]
    return filtered_data

def query_inverted(query_str: str, collection="default", n_results: int = None, max_distance = 0.75,
                   kwargs: Dict[str, Any] = {}, maximum_chars=4000, minimum_cross_encoder_score=0.05):
    """
    Query returns a dict with lists as values, this returns a list of dicts
    Also this supports maximum_chars as an alternative to n_results.
    Can also query multiple collections at once.

    It also flattens the metadatas into the main dict and renames elements to be more intuitive
    :param query_str:
    :param collection: string or list of strings. If list, it will query all collections and return the most relevant results.
    :param n_results: Maximum number of results to return. If specified with maximum_chars, this will set an alternative upper limit.
    :param max_distance:
    :param maximum_chars: When maximum chars is set it will return the minimum set of chunks that have in total less than maximum_chars characters.
    :param kwargs:
    :return:
    """
    # TODO good values for max_distance and minimum_cross_encoder_score
    if n_results is None and maximum_chars is None:
        raise ValueError("Either n_results or maximum_chars must be set")

    if maximum_chars and n_results is None:
        # assume 700 chars per chunk and account for variation
        n_results = maximum_chars//700 + 5
    if maximum_chars and n_results:
        n_results = max(n_results, maximum_chars//700 + 5)
    n_results_query = n_results
    if settings.USE_CROSS_ENCODER:
        n_results_query = n_results*4
    temp_n_results = n_results
    results_arr = []
    if isinstance(collection, list):
        n_results = n_results // len(collection)
        for col in collection:
            results = query(query_str, col, n_results_query, max_distance, kwargs)
            results_arr.append(results)
    else:
        results = query(query_str, collection, n_results_query, max_distance, kwargs)
        results_arr.append(results)
    flattened = []
    for results in results_arr:
        if not results or not any(results.values()):
            continue
        if "included" in results:
            del results["included"]
        length = len(results["ids"])
        if length == 0:
            continue

        inverted = []
        for i in range(length):
            entry = {}
            for key, value in results.items():
                if value:
                    entry[key] = value[i]
            inverted.append(entry)

        for i in range(length):
            inverted[i].update(inverted[i]["metadatas"])
            del inverted[i]["metadatas"]

        rename_dic = {"ids": "id", "documents": "content",  "distances": "distance", "embeddings": "embedding"}
        for entry in inverted:
            for key, value in rename_dic.items():
                if key in entry:
                    entry[value] = entry.pop(key)
        flattened.extend(inverted)
    if len(flattened) == 0:
        return []

    if settings.USE_CROSS_ENCODER:
        ragdb = RagDB()
        texts = [i["content"] for i in flattened]
        indices = ragdb.cross_encoder.rank(query_str, texts, return_documents=False, top_k=n_results)
        # TODO add a minimum cross encoder score
        indices = [i for i in indices if i["score"] > minimum_cross_encoder_score]
        flattened = [flattened[i["corpus_id"]] for i in indices]
    else:
        distance_sort_idx = np.argsort([i["distance"] for i in flattened])
        flattened = [flattened[i] for i in distance_sort_idx]

    if maximum_chars:
        cumsizes = np.cumsum([len(i["content"]) for i in flattened])
        maximum = np.searchsorted(cumsizes, maximum_chars)
        if temp_n_results:
            maximum = min(maximum, temp_n_results)
    else:
        maximum = temp_n_results
    flattened = flattened[:maximum]

    return flattened


def query_by_metadata(query_dict, collection="default", count: int = 5, **kwargs):
    """
    Query the ChromaDB for entries where metadata[key] == value

    For more advanced queries use the chroma client and the where statement directly
    See here https://docs.trychroma.com/guides#using-where-filters

    :param query_dict: a dictionary of key-value pairs to query for. E.g. if you search for images use {"type": "image"}
        and if you search for images from the document with id 123 use {"document_id": 123, "type": "image"}
    :param collection: the name of the collection to query
    :param n_results: the number of results to return
    """
    ragdb = RagDB()
    collection = ragdb.client.get_collection(name=collection)
    if len(query_dict) == 1:
        results = collection.get(where=query_dict, limit=count, **kwargs)
    else:
        new_query_dict = {"$and": []}
        for key, value in query_dict.items():
            new_query_dict["$and"].append({key: {"$eq": value}})
        results = collection.get(where=new_query_dict, limit=count, **kwargs)
    return results

def get_random_elements(count=5, collection="default"):
    """
    Get random elements from the ChromaDB

    :param count: the number of elements to return
    :param collection_name: the name of the collection to query
    """
    ragdb = RagDB()
    collection = ragdb.client.get_collection(name=collection)
    maximum = collection.count()
    start_index = random.randint(0, maximum - count)
    return collection.get(
        limit=count,
        offset=start_index
    )


def list_collections(raw=False):
    """
    List all collections in the db
    """
    ragdb = RagDB()
    if raw:
        return ragdb.client.list_collections()
    else:
        return [collection.name for collection in ragdb.client.list_collections()]


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
