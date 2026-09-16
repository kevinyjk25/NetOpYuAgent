"""Node adapter invariants; scripted replies are never semantic/model evidence."""
from pathlib import Path
import shutil
import subprocess

import pytest
from evaluation.dsh_shadow import _node_path


def test_dsh_terminal_adapter_node_invariants():
    node = shutil.which("node", path=_node_path())
    if not node:
        pytest.skip("Node required for the DSH adapter contract")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, "--test", "dsh-plugin-netopyu/tests/hybrid-terminal.test.js"],
                            cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
