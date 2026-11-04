"""Prompt construction and similarity-guarded RAG answers."""

import os

IDK_THRESHOLD = float(os.environ.get("IDK_THRESHOLD", "0.50"))


def build_rag_prompt(question: str, retrieved: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"Source [{hit['id']}]: {hit['text']}" for hit in retrieved
    )
    return [
        {"role": "system", "content": (
            "You are a knowledge-base assistant. Use ONLY the provided context to answer the question. "
            "If the context does not contain enough information, reply with 'I don't know.' "
            "Cite the source IDs you use in square brackets, for example [np_06]. "
            "Do not cite passage numbers."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above. Cite sources."},
    ]


def rag_answer_with_guard(retriever, generator, question: str, k: int = 5,
                          idk_threshold: float = IDK_THRESHOLD) -> dict:
    retrieved = retriever.retrieve(question, k=k)
    similarity = retrieved[0]["similarity"] if retrieved else 0.0
    result = {
        "question": question,
        "retrieved_ids": [hit["id"] for hit in retrieved],
        "top_similarity": similarity,
        "guarded": similarity < idk_threshold,
    }
    if result["guarded"]:
        result["answer"] = "I don't know. No relevant information found in the knowledge base."
    else:
        result["answer"] = generator.chat(build_rag_prompt(question, retrieved)).strip()
    return result
