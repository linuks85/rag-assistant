.PHONY: build run down logs eval clean

PYTHON ?= python3

build:
	docker compose build

run:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f rag-service

eval:
	$(PYTHON) eval.py

clean:
	docker compose down -v
	rm -rf chroma_rag evaluation.csv
