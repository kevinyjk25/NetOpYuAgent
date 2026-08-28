"""Enterprise service-system simulation exposed through real MCP processes.

The data backend is deterministic SQLite.  The protocol boundary is not a
mock: DSH/Hermes connect to the servers over MCP stdio or Streamable HTTP.
"""

from .store import ServiceStore, default_store_path

__all__ = ["ServiceStore", "default_store_path"]
