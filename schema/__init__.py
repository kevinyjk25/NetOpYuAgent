"""
schema/ — unified tool argument schema framework.

Provides a JSON-Schema-subset for describing tool arguments, with:

  - Validate-and-coerce  : turn whatever the LLM produces into the canonical shape
  - Multi-source import  : MCP inputSchema, OpenAPI parameters/requestBody, dict-style metadata
  - Prompt rendering     : auto-generate the "Args:" line shown to the LLM
  - Diagnostic errors    : tell the LLM exactly what was wrong, in its own format

Why JSON Schema (subset):
  - MCP defines tool args as JSON Schema (inputSchema field) — drop-in compatible
  - OpenAPI 3 defines parameters / requestBody as JSON Schema — drop-in compatible
  - Most LLMs already understand JSON Schema natively (function calling)

We intentionally support a *subset* of JSON Schema (not all of draft-07/2020-12)
to keep validation fast and the framework easy to reason about.
"""
from .types     import (
    ArgSchema, ArgField, FieldType, CoerceResult, ValidationError,
)
from .validator import validate_and_coerce
from .importers import (
    from_dict_metadata, from_mcp_input_schema, from_openapi_operation,
)
from .prompt    import render_args_for_prompt
from .registry  import SchemaRegistry, get_schema_registry

__all__ = [
    "ArgSchema", "ArgField", "FieldType", "CoerceResult", "ValidationError",
    "validate_and_coerce",
    "from_dict_metadata", "from_mcp_input_schema", "from_openapi_operation",
    "render_args_for_prompt",
    "SchemaRegistry", "get_schema_registry",
]
