"""Ollama client for grounded answer generation."""

import os

import ollama
import requests


def check_ollama(host: str, model: str) -> None:
    """Fail early with an actionable message when Ollama is unavailable."""
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach Ollama at {host}; start Ollama and check OLLAMA_HOST") from exc
    names = {
        item.get("name") or item.get("model")
        for item in response.json().get("models", [])
    }
    if model not in names:
        raise RuntimeError(f"Ollama model {model!r} is missing; run `ollama pull {model}`")


class LLMClient:
    def __init__(self, model: str = "llama3.2:3b", host: str | None = None) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        check_ollama(self.host, self.model)
        self.client = ollama.Client(host=self.host)

    def generate(self, prompt: str, format: str | None = None, **options) -> str:
        settings = {"temperature": 0.0, "seed": 42, **options}
        response = self.client.generate(model=self.model, prompt=prompt, format=format, options=settings)
        return response["response"]

    def chat(self, messages: list[dict], format: str | None = None, **options) -> str:
        settings = {"temperature": 0.0, "seed": 42, **options}
        response = self.client.chat(model=self.model, messages=messages, format=format, options=settings)
        return response["message"]["content"]
