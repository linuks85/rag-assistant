"""Evaluate the running service against the raid's fifteen gold questions."""

import csv
import os
import sys
import time

import httpx

from corpus import GOLD_SET

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


def wait_for_service(timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Service at {BASE_URL} not ready after {timeout:g}s")


def evaluate_response(question: str, expected_doc: str, required_kw: list[str], data: dict) -> dict:
    ids = data["retrieved_ids"]
    answer = data["answer"]
    return {
        "question": question,
        "expected_doc": expected_doc,
        "retrieved_top": ids[0] if ids else "",
        "retrieved_ids": ",".join(ids),
        "retrieval_hit": expected_doc in ids[:5],
        "keyword_hit": all(keyword.lower() in answer.lower() for keyword in required_kw),
        "guarded": bool(data["guarded"]),
        "top_similarity": float(data["top_similarity"]),
        "answer": answer,
    }


def run_eval() -> tuple[float, float]:
    print(f"Hitting {BASE_URL} ...")
    wait_for_service(timeout=60)
    rows = []
    with httpx.Client(timeout=120.0) as client:
        for question, expected_doc, required_kw in GOLD_SET:
            response = client.post(f"{BASE_URL}/query", json={"question": question, "k": 5})
            response.raise_for_status()
            rows.append(evaluate_response(question, expected_doc, required_kw, response.json()))
    with open("evaluation.csv", "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    recall_at_5 = sum(row["retrieval_hit"] for row in rows) / len(rows)
    faithfulness = sum(row["keyword_hit"] for row in rows) / len(rows)
    guarded_count = sum(row["guarded"] for row in rows)
    print(f"recall@5: {recall_at_5:.4f} ({sum(row['retrieval_hit'] for row in rows)}/{len(rows)})")
    print(f"keyword_hit: {faithfulness:.4f} ({sum(row['keyword_hit'] for row in rows)}/{len(rows)})")
    print(f"guarded: {guarded_count}/{len(rows)}")
    print("saved to: evaluation.csv")
    print(f"audit floor:  recall@5 >= 0.5  →  {'PASS' if recall_at_5 >= 0.5 else 'FAIL'}")
    return recall_at_5, faithfulness


if __name__ == "__main__":
    recall_at_5, faithfulness = run_eval()
    sys.exit(0 if recall_at_5 >= 0.5 else 1)
