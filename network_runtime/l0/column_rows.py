"""Bounded column-metadata projection, not a query, filter, loop language or sanitizer.

All rows are decoded or the operation fails. No truncation, coercion, guessed
indices or defaults. Response status/completeness and publication policy belong
to the surrounding domain contract, not this source-independent data primitive.
"""

from .structured_schema import DataBindingError, join_pointer, schema_location, schema_types, snapshot_json


def projection_options(fields, max_rows, max_columns):
    if (not isinstance(fields, list) or not 1 <= len(fields) <= 32
            or any(not isinstance(f, str) or not f.strip() or len(f) > 128 for f in fields)
            or len(set(fields)) != len(fields)):
        raise DataBindingError("column_fields", "", "one to 32 distinct explicit field names required")
    if type(max_rows) is not int or not 1 <= max_rows <= 256:
        raise DataBindingError("column_row_budget", "", "row bound must be an integer from one to 256")
    if type(max_columns) is not int or not 1 <= max_columns <= 128:
        raise DataBindingError("column_metadata_budget", "", "column bound must be an integer from one to 128")


def check_projection_schema(source, pointer, target, target_pointer, fields, max_rows, max_columns):
    """Structural compatibility only. Every selected cell is checked again later."""
    projection_options(fields, max_rows, max_columns)
    schema_location(source, pointer)  # Validate the pointer before concatenation.
    columns_path, rows_path = join_pointer(pointer, "columns"), join_pointer(pointer, "data")
    columns, _ = schema_location(source, columns_path)
    rows, _ = schema_location(source, rows_path)
    row, _ = schema_location(source, rows_path + "/0")
    output, _ = schema_location(target, target_pointer)
    item, _ = schema_location(target, target_pointer + "/0")
    if (schema_types(columns) != {"object"} or schema_types(rows) != {"array"}
            or schema_types(row) != {"array"} or schema_types(output) != {"array"}
            or schema_types(item) != {"object"} or set(item.get("properties", {})) != set(fields)
            or set(item.get("required", [])) != set(fields) or item.get("additionalProperties") is not False):
        raise DataBindingError("column_projection_shape", target_pointer, "explicit columns, row arrays and exact selected output fields required")
    for field in fields:
        index, _ = schema_location(source, join_pointer(columns_path, field) + "/index")
        if schema_types(index) != {"integer"}:
            raise DataBindingError("column_index_schema", columns_path, "column index needs an integer schema")


def decode_column_rows(value, fields, *, max_rows, max_columns):
    projection_options(fields, max_rows, max_columns)
    value = snapshot_json(value)
    if not isinstance(value, dict) or not isinstance(value.get("columns"), dict) or not isinstance(value.get("data"), list):
        raise DataBindingError("column_envelope", "", "columns object and data array required")
    columns, rows = value["columns"], value["data"]
    if not 1 <= len(columns) <= max_columns:
        raise DataBindingError("column_metadata_budget", "/columns", "metadata is empty or exceeds the explicit bound")
    if len(rows) > max_rows:
        raise DataBindingError("column_row_budget", "/data", "all rows must fit the explicit bound; truncation is forbidden")
    indices = {}
    for field, metadata in columns.items():
        at = join_pointer("/columns", field) + "/index"
        index = metadata.get("index") if isinstance(metadata, dict) else None
        if type(index) is not int or not 0 <= index < max_columns:
            raise DataBindingError("column_index", at, "bounded nonnegative integer index required")
        if index in indices.values():
            raise DataBindingError("column_index_duplicate", at, "column indices must be distinct, including unselected fields")
        indices[field] = index
    if set(fields) - columns.keys():
        raise DataBindingError("column_missing", "/columns", "a selected field has no metadata")
    width = max(indices.values()) + 1
    result = []
    for number, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != width:
            raise DataBindingError("column_row_width", join_pointer("/data", number), "every row must match the declared indexed width")
        result.append({field: row[indices[field]] for field in fields})
    return result
