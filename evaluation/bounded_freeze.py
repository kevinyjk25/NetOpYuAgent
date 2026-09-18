"""Content manifests for the declared local R0 execution dependency closure.

Hash entire selected code/package trees, not only executable entry points.
No package installation, model loading, source-script execution or Git mutation.
OS/kernel services remain a declared host trust boundary, not an attestation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import sys
import sysconfig

from evaluation.bounded_pilot import ROOT, checked_seal, seal
from evaluation.dsh_support import _default_dsh_binary, _node_path

EXCLUDED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def tree_manifest(root, *, declared_roots=()):
    root = Path(root).absolute()
    if not root.is_dir():
        raise ValueError("execution dependency tree missing")
    files = {}
    links = {}
    allowed = (root.resolve(), *(Path(path).resolve() for path in declared_roots))
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = sorted(name for name in children if name not in EXCLUDED_DIRS)
        # pnpm uses internal directory links extensively. Record the edge;
        # contents must be covered by this tree or another declared tree.
        for name in children:
            path = Path(directory) / name
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if not any(resolved.is_relative_to(base) for base in allowed):
                    raise ValueError(f"declare symlinked dependency directory separately: {path}")
                links[str(path.relative_to(root))] = {"link_target": os.readlink(path),
                                                      "resolved": str(resolved)}
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if not path.is_file():
                raise ValueError("nonregular dependency file")
            files[str(path.relative_to(root))] = {
                "sha256": digest_file(path), "bytes": path.stat().st_size,
                "link_target": os.readlink(path) if path.is_symlink() else None}
    if not files:
        raise ValueError("empty execution dependency tree")
    return {"root": str(root), "files": files, "directory_links": links}


def freeze(trees, files, *, metadata=None):
    if not trees or not files:
        raise ValueError("explicit dependency trees and executable files required")
    return seal({"schema": "ensuredskill.io/bounded-execution-freeze/v1",
        "trees": {name: tree_manifest(path, declared_roots=trees.values())
                  for name, path in sorted(trees.items())},
        "files": {name: {"path": str(Path(path).absolute()), "resolved": str(Path(path).resolve()),
                          "sha256": digest_file(path)} for name, path in sorted(files.items())},
        "metadata": metadata or {}, "actualModelCalls": 0,
        "host_boundary": "OS/kernel/system services trusted; no complete machine or build attestation",
        "bytecode_policy": "excluded caches; isolated no-write PYTHONPYCACHEPREFIX required",
        "excluded_directories": sorted(EXCLUDED_DIRS)})


def verify(snapshot):
    value = checked_seal(snapshot)
    rebuilt = freeze({name: tree["root"] for name, tree in value["trees"].items()},
                     {name: file["path"] for name, file in value["files"].items()},
                     metadata=value["metadata"])
    if rebuilt != value:
        raise ValueError("execution dependency content, inventory or link drift")
    return {"unchanged": True, "digest": value["digest"], "actualModelCalls": 0}


def local_spec(*, assets, codec, source_root=ROOT):
    """Selected full package trees and all project execution code/config.

    Runtime generated artifacts, source labels and Provider inputs are frozen
    by their separate protocol/source/reference manifests, not copied here.
    """
    root = Path(source_root)
    folders = ("evaluation", "skill_authoring", "network_runtime", "effect_runtime",
               "dsh_adapter", "dsh-plugin-netopyu", "service_mcp", "network_mcp",
               "runtime", "skills", "tools")
    trees = {"project/" + name: root / name for name in folders if (root / name).is_dir()}
    dsh = _default_dsh_binary()
    modules = next(parent for parent in dsh.parents if parent.name == "node_modules")
    trees.update({"installed_dsh_node_modules": modules,
                  "python_site_packages": Path(sysconfig.get_path("purelib")),
                  "python_standard_library": Path(sysconfig.get_path("stdlib"))})
    node = shutil.which("node", path=_node_path())
    if node is None:
        raise ValueError("installed Node executable missing")
    files = {"python": Path(sys.executable), "node": Path(node), "dsh": dsh,
             "parser_codec": Path(codec["path"])}
    # macOS framework Python's executable is only a launcher; the interpreter
    # implementation lives outside the stdlib/package trees.
    framework = Path(sys.base_prefix) / "Python"
    if sys.platform == "darwin" and framework.is_file():
        files["python_framework"] = framework
    for path in sorted(root.glob("requirements*.txt")):
        files[path.name] = path
    for name in ("pyproject.toml", "pytest.ini"):
        if (root / name).is_file():
            files[name] = root / name
    pins = assets.inspect()
    for name in ("renderer", "tokenizer", "model"):
        files["preflight/" + name] = Path(pins[name]["path"])
    for index, pin in enumerate(pins["dependencies"]):
        files[f"preflight/dependency/{index}"] = Path(pin["path"])
    return trees, files, {"platform": platform.platform(), "python": sys.version,
        "assets_digest": pins["digest"], "parser_pin": codec,
        "scope": "selected project execution trees + entire installed Python/DSH packages + named native assets",
        "research_clean_git_freeze": False, "purpose": "R0 zero-inference engineering acceptance"}
