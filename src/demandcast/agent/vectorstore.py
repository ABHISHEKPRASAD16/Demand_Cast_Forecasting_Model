"""Persisted Chroma vector store over the RAG knowledge base.

Embeddings are computed locally via sentence-transformers - no API key or
network call needed to build or query the vector store. Only the agent's
own LLM reasoning (agent.py) needs an Anthropic key.
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from demandcast.agent.knowledge import build_knowledge_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PERSIST_DIR = Path(__file__).resolve().parents[3] / "data" / "vectorstore"
COLLECTION_NAME = "demandcast_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(persist_directory: Path = PERSIST_DIR) -> Chroma:
    """(Re)build the vector store from scratch and persist it to disk."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )

    # start clean each time so documents removed/changed upstream don't linger
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    documents = build_knowledge_base()
    vectorstore.add_documents(documents)
    logger.info("Indexed %d documents into %s", len(documents), persist_directory)
    return vectorstore


def load_vectorstore(persist_directory: Path = PERSIST_DIR) -> Chroma:
    """Load the already-built vector store without recomputing embeddings."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def main() -> None:
    build_vectorstore()


if __name__ == "__main__":
    main()
