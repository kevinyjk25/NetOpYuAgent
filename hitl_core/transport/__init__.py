"""
hitl_core.transport — Optional I/O adapters.

The core hitl_core package has zero web framework / SSE dependencies.
These adapters add convenient HTTP + SSE wiring on top, but are 100%
optional — hosts can skip them and roll their own transport layer.

Lazy imports here so importing `hitl_core.transport` doesn't pull
fastapi unless the host actually uses it.
"""

# We don't import the modules here to keep the import cheap. Hosts
# explicitly:
#
#   from hitl_core.transport.http_adapter import build_http_router
#   from hitl_core.transport.sse_adapter  import stream_pipeline_as_sse
#
# This way `import hitl_core.transport` doesn't error on missing
# fastapi when the host is going to use only the SSE adapter (or
# vice versa).

__all__: list[str] = []
