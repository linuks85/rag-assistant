# RAG Document Assistant

Question answering over a small corpus of 40 technical snippets (NumPy, Pandas, ML, NLP).
Retrieval with **ChromaDB** and **sentence-transformers** (`all-MiniLM-L6-v2`), answer generation with a **local LLM**
(Ollama, `llama3.2:3b`), served by **FastAPI** and packaged with **Docker Compose**.

> **Team project.** Built in a pair at Tomorrow School (AI Piscine). Nurlan Tleukin: ingestion and the service
> (retriever, RAG logic, API, Docker). Assel Kuketayeva: evaluation and report. The code was written with AI-assistant help
> as part of the learning process. The corpus and test questions are small and partly synthetic, so the metrics below show
> the evaluation method, not production quality.

## How it works

1. On startup the 40 snippets are embedded and indexed in ChromaDB (cosine distance).
2. `POST /query` retrieves the top-k snippets (default 5).
3. **Hallucination guard:** if the best similarity is below a threshold (default 0.50), the service answers "I don't know" and does not call the LLM.
4. Otherwise the context goes to the local LLM; the response includes the source IDs, top similarity and a `guarded` flag.

## Evaluation

`make eval` runs 15 gold questions and reports recall@5, keyword faithfulness and the number of guarded answers.
The threshold was chosen by a sweep over four values (see [REPORT.md](REPORT.md)).
Result on the gold set: **recall@5 = 1.0 (15/15)**; an out-of-corpus question ("population of Mars") got similarity 0.177 and was correctly guarded.
The gold set is small, so this does not prove the threshold is optimal for other queries.

## Run

Requirements: Docker with Compose v2 and [Ollama](https://ollama.com) running locally.

```bash
ollama pull llama3.2:3b
make run          # build and start on http://localhost:8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I solve a linear system in NumPy?"}'
```

`make eval` needs Python 3.10+ with `requests` and `pandas` installed (`pip install -r requirements.txt`).
Other commands: `make logs`, `make down`, `make clean`. Change the threshold: `IDK_THRESHOLD=0.40 make run`.

## Structure

- `corpus.py` corpus and gold questions · `ingest.py` chunking with overlap
- `app/retriever.py` embeddings + ChromaDB · `app/llm_client.py` Ollama client
- `app/rag.py` prompt and guard · `app/main.py` FastAPI (`/health`, `/query`)
- `eval.py` metrics · `REPORT.md` architecture, sweep, audit status

## Stack

Python, FastAPI, ChromaDB, sentence-transformers, Ollama, Docker.
