"""
agent_memory/retrieval/vector_store.py

Thread-safe TF-IDF index with LRU eviction.
Zero external dependencies — uses only Python stdlib.
Swappable: replace with sentence-transformers by subclassing BaseVectorStore.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class TFIDFIndex:
    """
    Thread-safe in-memory TF-IDF index with LRU eviction.

    Args:
        max_docs: Maximum documents before LRU eviction kicks in.
                  Defaults to 50_000. Set to 0 for unlimited (not recommended).
    """

    def __init__(self, max_docs: int = 50_000) -> None:
        self._max_docs = max_docs
        # Use OrderedDict for LRU tracking (doc_id → {token: tf})
        self._tf: OrderedDict[str, Dict[str, float]] = OrderedDict()
        # token → set of doc_ids (for IDF)
        self._idf_raw: Dict[str, set] = defaultdict(set)
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────────────────────────

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        if not tokens:
            return
        counts: Dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        total = len(tokens)
        tf = {t: c / total for t, c in counts.items()}

        with self._lock:
            # Remove existing entry if updating
            if doc_id in self._tf:
                self._remove_locked(doc_id)
            # Evict LRU if over capacity
            if self._max_docs > 0 and len(self._tf) >= self._max_docs:
                evicted_id, _ = self._tf.popitem(last=False)
                self._cleanup_idf_locked(evicted_id)
                logger.debug("TF-IDF evicted LRU doc %s", evicted_id)
            # Insert
            self._tf[doc_id] = tf
            for t in tf:
                self._idf_raw[t].add(doc_id)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._remove_locked(doc_id)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return list of (doc_id, score) sorted by score desc."""
        tokens = _tokenize(text)
        if not tokens:
            return []
        with self._lock:
            if not self._tf:
                return []
            N = len(self._tf)
            scores: Dict[str, float] = defaultdict(float)
            for t in tokens:
                docs_with_t = self._idf_raw.get(t)
                if not docs_with_t:
                    continue
                idf = math.log((N + 1) / (len(docs_with_t) + 1)) + 1.0
                for doc_id, tf_map in self._tf.items():
                    if t in tf_map:
                        scores[doc_id] += tf_map[t] * idf
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._tf)

    def clear(self) -> None:
        with self._lock:
            self._tf.clear()
            self._idf_raw.clear()

    # ── private ──────────────────────────────────────────────────────────────

    def _remove_locked(self, doc_id: str) -> None:
        tf = self._tf.pop(doc_id, None)
        if tf:
            self._cleanup_idf_locked(doc_id)

    def _cleanup_idf_locked(self, doc_id: str) -> None:
        empty_tokens = []
        for token, doc_set in self._idf_raw.items():
            doc_set.discard(doc_id)
            if not doc_set:
                empty_tokens.append(token)
        for t in empty_tokens:
            del self._idf_raw[t]
