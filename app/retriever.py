"""Persistent semantic retrieval over the raid corpus."""

import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:
    """ChromaDB collection encoded with all-MiniLM-L6-v2."""

    def __init__(self, path: str = "./chroma_rag", model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "knowledge_base") -> None:
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def index(self, docs: list[tuple[str, str]]) -> None:
        """Replace the indexed documents with the supplied ID/text pairs."""
        ids = [doc_id for doc_id, _ in docs]
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique")
        existing = self.collection.get()["ids"]
        if existing:
            self.collection.delete(ids=existing)
        if not docs:
            return
        texts = [text for _, text in docs]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Return nearest documents with cosine similarity in descending order."""
        if k < 1:
            raise ValueError("k must be positive")
        count = self.collection.count()
        if count == 0:
            return []
        embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()
        result = self.collection.query(
            query_embeddings=embedding, n_results=min(k, count), include=["documents", "distances"]
        )
        return [
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "similarity": 1.0 - result["distances"][0][i],
            }
            for i in range(len(result["ids"][0]))
        ]
