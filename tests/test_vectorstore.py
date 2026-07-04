from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from demandcast.agent import vectorstore


class _FakeEmbeddings(Embeddings):
    """Deterministic, instant embeddings - avoids downloading/running the
    real sentence-transformers model in tests, which only need to exercise
    the Chroma read/write/rebuild logic, not embedding quality."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_build_vectorstore_indexes_all_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr(
        vectorstore,
        "build_knowledge_base",
        lambda: [
            Document(page_content="doc one", metadata={"source": "a"}),
            Document(page_content="doc two", metadata={"source": "b"}),
        ],
    )

    vs = vectorstore.build_vectorstore(persist_directory=tmp_path)

    assert len(vs.get()["ids"]) == 2


def test_load_vectorstore_reads_back_persisted_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr(
        vectorstore,
        "build_knowledge_base",
        lambda: [Document(page_content="doc one", metadata={"source": "a"})],
    )
    vectorstore.build_vectorstore(persist_directory=tmp_path)

    reloaded = vectorstore.load_vectorstore(persist_directory=tmp_path)

    assert len(reloaded.get()["ids"]) == 1
    assert reloaded.get()["documents"] == ["doc one"]


def test_rebuilding_clears_stale_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: _FakeEmbeddings())

    monkeypatch.setattr(
        vectorstore,
        "build_knowledge_base",
        lambda: [
            Document(page_content="doc one", metadata={"source": "a"}),
            Document(page_content="doc two", metadata={"source": "b"}),
        ],
    )
    vectorstore.build_vectorstore(persist_directory=tmp_path)

    monkeypatch.setattr(
        vectorstore,
        "build_knowledge_base",
        lambda: [Document(page_content="doc three", metadata={"source": "c"})],
    )
    vs = vectorstore.build_vectorstore(persist_directory=tmp_path)

    assert vs.get()["documents"] == ["doc three"]
