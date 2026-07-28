"""ChromaDB singleton client."""

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from config.settings import settings

COLLECTION_CHUNKS = "paper_chunks"
COLLECTION_META = "paper_meta"


@lru_cache
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_collection(name: str = COLLECTION_CHUNKS) -> Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_chunks_collection() -> Collection:
    return get_collection(COLLECTION_CHUNKS)


def get_meta_collection() -> Collection:
    return get_collection(COLLECTION_META)


def paper_exists(paper_id: str) -> bool:
    coll = get_chunks_collection()
    result = coll.get(where={"paper_id": paper_id}, limit=1)
    return bool(result and result.get("ids"))
