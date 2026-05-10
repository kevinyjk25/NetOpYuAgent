"""
retrieval package — pluggable retrievers and meta-tool framework.

Provides:
  - Retriever interface (sync + async)
  - HybridRetriever  : BM25 + embedding fusion (default)
  - BM25Retriever    : pure lexical, fallback
  - KeywordRetriever : legacy word-overlap, last resort

  - MetaToolRegistry : runtime registration of "always-injected" meta tools
                       (list_tools, list_skills, search_memory, …)

  Both retrievers and meta-tools are config-driven and can be swapped without
  touching the runtime loop.
"""
from .base       import Retriever, RetrievalResult, Match
from .bm25       import BM25Retriever
from .keyword    import KeywordRetriever
from .embedding  import EmbeddingRetriever
from .hybrid     import HybridRetriever
from .cache      import CachedRetriever
from .llm_judge  import LLMJudgeRetriever
from .meta_tool  import MetaTool, MetaToolRegistry, get_meta_tool_registry
from .meta_tool  import (
    make_list_tools_meta_tool, make_list_skills_meta_tool, make_tool_details_meta_tool,
)
from .factory    import (
    build_retriever, build_tool_retriever, build_skill_retriever,
    build_tool_retriever_async, build_skill_retriever_async,
    tools_to_corpus, skills_to_corpus,
)

__all__ = [
    "Retriever", "RetrievalResult", "Match",
    "BM25Retriever", "KeywordRetriever", "EmbeddingRetriever", "HybridRetriever",
    "CachedRetriever", "LLMJudgeRetriever",
    "MetaTool", "MetaToolRegistry", "get_meta_tool_registry",
    "make_list_tools_meta_tool", "make_list_skills_meta_tool", "make_tool_details_meta_tool",
    "build_retriever", "build_tool_retriever", "build_skill_retriever",
    "build_tool_retriever_async", "build_skill_retriever_async",
    "tools_to_corpus", "skills_to_corpus",
]
