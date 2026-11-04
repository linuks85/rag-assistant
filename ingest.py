"""Optional word-based chunking for documents outside the seed corpus."""

from corpus import SEED_CORPUS


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into word chunks with a shared boundary between neighbors."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_full_corpus(
    external_docs: list[tuple[str, str]] | None = None,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[tuple[str, str]]:
    """Return the seed corpus plus numbered chunks from external documents."""
    corpus = list(SEED_CORPUS)
    for doc_id, text in external_docs or []:
        for number, chunk in enumerate(chunk_text(text, chunk_size, overlap), start=1):
            corpus.append((f"{doc_id}_{number:03d}", chunk))
    ids = [doc_id for doc_id, _ in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("document IDs must be unique")
    return corpus
