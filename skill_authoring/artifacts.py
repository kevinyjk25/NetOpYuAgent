from pathlib import Path
import json
from network_runtime.l0.structured_schema import MAX_BYTES
from network_runtime.l0.read_contracts import _source_object
def read_json(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("input JSON exceeds byte budget")
    return _source_object(path.read_text(encoding="utf-8"))


def write_artifacts(output: str | Path, files: dict) -> None:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        (root / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
