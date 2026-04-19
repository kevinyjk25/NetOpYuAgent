"""
integrations/embedder.py
-------------------------
Real embedding backends — used by BOTH mock and pragmatic modes.

Backends:
  OllamaEmbedder  — nomic-embed-text, mxbai-embed-large via local Ollama
  OpenAIEmbedder  — text-embedding-3-small / large via OpenAI API
  StubEmbedder    — deterministic hash (fallback / none backend)

Usage (injected by main.py):
    from integrations.embedder import build_embedder
    embedder = build_embedder(cfg.embeddings)
    ingestion_pipeline.set_embedder(embedder)
"""
from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


class StubEmbedder:
    """Deterministic hash stub — zero deps, always works."""
    def __init__(self, dim: int = 768) -> None:
        self.DIM = dim

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        base   = [b / 255.0 for b in digest]
        vec    = (base * (self.DIM // len(base) + 1))[: self.DIM]
        norm   = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class OllamaEmbedder:
    """Ollama /api/embeddings endpoint. Falls back to StubEmbedder on error."""
    def __init__(self, model: str, base_url: str, dim: int) -> None:
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self.DIM      = dim

    async def embed(self, text: str) -> list[float]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                resp.raise_for_status()
                vec = resp.json().get("embedding", [])
                if not vec:
                    raise ValueError("Empty embedding")
                norm = sum(v * v for v in vec) ** 0.5 or 1.0
                return [v / norm for v in vec]
        except Exception as exc:
            logger.warning("OllamaEmbedder: failed (%s), falling back to stub", exc)
            return await StubEmbedder(self.DIM).embed(text)


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small / large."""
    def __init__(self, model: str, dim: int, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model    = model
        self.DIM      = dim
        self._api_key = os.getenv(api_key_env, "")

    async def embed(self, text: str) -> list[float]:
        try:
            from openai import AsyncOpenAI
            resp = await AsyncOpenAI(api_key=self._api_key).embeddings.create(
                model=self.model, input=text
            )
            vec  = resp.data[0].embedding
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            return [v / norm for v in vec]
        except Exception as exc:
            logger.warning("OpenAIEmbedder: failed (%s), falling back to stub", exc)
            return await StubEmbedder(self.DIM).embed(text)


def build_embedder(emb_cfg):
    """Factory from EmbeddingsConfig."""
    backend = emb_cfg.backend.lower()
    if backend == "ollama":
        logger.info("Embedder: Ollama %s dim=%d", emb_cfg.model, emb_cfg.dim)
        return OllamaEmbedder(model=emb_cfg.model, base_url=emb_cfg.base_url, dim=emb_cfg.dim)
    if backend == "openai":
        logger.info("Embedder: OpenAI %s dim=%d", emb_cfg.model, emb_cfg.dim)
        return OpenAIEmbedder(model=emb_cfg.model, dim=emb_cfg.dim)
    logger.info("Embedder: stub (dim=%d)", emb_cfg.dim)
    return StubEmbedder(dim=emb_cfg.dim)
