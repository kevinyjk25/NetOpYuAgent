from pathlib import Path

import pytest

from evaluation.bounded_freeze import freeze, tree_manifest, verify


def specimen(tmp_path):
    tree = tmp_path / "code"
    tree.mkdir()
    (tree / "entry.py").write_text("pass\n")
    binary = tmp_path / "executable"
    binary.write_text("not executable fixture")
    return tree, binary, freeze({"code": tree}, {"executable": binary})


def test_full_inventory_detects_non_entry_dependency_and_new_file(tmp_path):
    tree, _, snapshot = specimen(tmp_path)
    assert verify(snapshot)["unchanged"] is True
    (tree / "indirect.py").write_text("new dependency")
    with pytest.raises(ValueError, match="drift"):
        verify(snapshot)


def test_file_and_symlink_target_drift_rejected(tmp_path):
    tree, binary, _ = specimen(tmp_path)
    alias = tree / "alias"
    alias.symlink_to(binary)
    snapshot = freeze({"code": tree}, {"executable": binary})
    assert verify(snapshot)["unchanged"]
    binary.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        verify(snapshot)


def test_bytecode_is_explicitly_excluded_not_execution_source(tmp_path):
    tree, _, snapshot = specimen(tmp_path)
    cache = tree / "__pycache__"
    cache.mkdir()
    (cache / "entry.pyc").write_bytes(b"fixture")
    assert verify(snapshot)["unchanged"]
    assert "PYTHONPYCACHEPREFIX" in snapshot["bytecode_policy"]


def test_directory_links_require_explicit_tree(tmp_path):
    tree, _, _ = specimen(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tree / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="separately"):
        tree_manifest(tree)


def test_snapshot_does_not_claim_clean_research_or_machine_attestation(tmp_path):
    _, _, snapshot = specimen(tmp_path)
    assert snapshot["actualModelCalls"] == 0
    assert "no complete machine" in snapshot["host_boundary"]
    assert Path(snapshot["files"]["executable"]["path"]).is_absolute()


def test_internal_package_directory_links_have_content_and_edge_pins(tmp_path):
    tree, binary, _ = specimen(tmp_path)
    package = tree / "store" / "package"
    package.mkdir(parents=True)
    (package / "main.js").write_text("export default 1;")
    (tree / "package").symlink_to(package, target_is_directory=True)
    snapshot = freeze({"code": tree}, {"binary": binary})
    assert "package" in snapshot["trees"]["code"]["directory_links"]
    assert "store/package/main.js" in snapshot["trees"]["code"]["files"]
    assert verify(snapshot)["unchanged"]


def test_local_spec_includes_indirect_project_packages_and_framework(monkeypatch, tmp_path):
    from evaluation import bounded_freeze as module

    for name in ("evaluation", "runtime", "skills", "tools"):
        (tmp_path / name).mkdir()
    framework = tmp_path / "interpreter"
    framework.mkdir()
    (framework / "Python").write_bytes(b"interpreter-fixture")
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.sys, "base_prefix", str(framework))
    monkeypatch.setattr(module, "_default_dsh_binary", lambda: tmp_path / "node_modules/dsh/bin.js")
    monkeypatch.setattr(module.shutil, "which", lambda *args, **kwargs: str(tmp_path / "node"))

    class AssetsFixture:
        def inspect(self):
            return {**{name: {"path": str(tmp_path / name)} for name in ("renderer", "tokenizer", "model")},
                    "dependencies": [], "digest": "fixture"}

    trees, files, _ = module.local_spec(assets=AssetsFixture(), codec={"path": str(tmp_path / "codec")},
                                       source_root=tmp_path)
    assert {"project/runtime", "project/skills", "project/tools"} <= trees.keys()
    assert files["python_framework"] == framework / "Python"
