"""Finite local vocab-only acceptance; no service, context, or inference call.

Outputs are new-only probe artifacts, not official pilot or model-quality data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(helper, model, output):
    helper, model, output = Path(helper).resolve(), Path(model).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = [
        ("empty", b"", True),
        ("english", b"Hello world", True),
        ("english_repeat", b"Hello world", True),
        ("unicode", "你好，世界。\nLine two\n🙂".encode(), True),
        ("special", b"<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", True),
        ("nul", b"A\x00B", True),
        ("nul_prefix", b"A", True),
        ("invalid_utf8", b"\xff", False),
        ("prompt_limit", b"a" * (4 * 1024 * 1024 + 1), False),
        ("token_limit", b"a " * 262145, False),
    ]
    rows, encoded, loaded = [], {}, set()
    for name, prompt, accepted in fixtures:
        started = time.monotonic()
        try:
            result = subprocess.run(
                [str(helper), "--model", str(model)],
                input=prompt, capture_output=True, timeout=45,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "DYLD_PRINT_LIBRARIES": "1"},
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            (output / (name + ".stdout")).write_bytes(error.stdout or b"")
            (output / (name + ".stderr")).write_bytes(error.stderr or b"")
            (output / "failure.json").write_text(json.dumps({
                "fixture": name, "reason": "helper_timeout_45s", "actualModelCalls": 0,
                "all_expected_decisions_correct": False,
            }) + "\n")
            raise
        (output / (name + ".stdout")).write_bytes(result.stdout)
        (output / (name + ".stderr")).write_bytes(result.stderr)
        log = result.stderr.decode("utf-8", errors="replace")
        loaded.update(re.findall(r"(/Applications/Ollama\.app/Contents/Resources/[^\n]*\.dylib)", log))
        assert len(result.stdout) <= 64 * 1024 * 1024
        row = {"name": name, "input_sha256": hashlib.sha256(prompt).hexdigest(),
               "input_bytes": len(prompt), "expected_accept": accepted, "returncode": result.returncode,
               "elapsed_seconds": time.monotonic() - started}
        if accepted:
            assert result.returncode == 0, (name, log[-2000:])
            value = json.loads(result.stdout)
            assert set(value) == {"token_ids", "count", "add_special", "parse_special"}
            assert value["count"] == len(value["token_ids"]) <= 262144
            assert value["add_special"] is True and value["parse_special"] is True
            assert all(type(token) is int and token >= 0 for token in value["token_ids"])
            assert "vocab only - skipping tensors" in log
            assert "no backend_init/context/decode/warmup" in log
            encoded[name] = value["token_ids"]
            row.update(count=value["count"], vocab_skip_log=True)
        else:
            assert result.returncode == 2 and not result.stdout, name
            assert "tokenizer: rejected:" in log
            if name in {"invalid_utf8", "prompt_limit"}:
                assert "vocab only - skipping tensors" not in log
        rows.append(row)
    assert encoded["english"] == encoded["english_repeat"]
    assert encoded["nul"] != encoded["nul_prefix"] and len(encoded["nul"]) > len(encoded["nul_prefix"])
    imports = subprocess.check_output(["/usr/bin/nm", "-u", str(helper)], text=True)
    (output / "helper-imports.txt").write_text(imports)
    actual_llama_imports = set(re.findall(r"\b(_llama_\w+)", imports))
    assert actual_llama_imports == {
        "_llama_log_set", "_llama_model_default_params", "_llama_model_load_from_file",
        "_llama_model_get_vocab", "_llama_model_free", "_llama_tokenize",
    }
    assert loaded, "dyld must report actual Ollama dynamic-library paths"
    dependencies = [{"loaded_path": path, "real_path": str(Path(path).resolve()), "sha256": sha256(path)}
                    for path in sorted(loaded)]
    assert any(Path(item["real_path"]).name == "libllama.0.3.0.dylib" for item in dependencies)
    pinned = helper.parent / "pinned"
    sources = [{"relative_path": str(path.relative_to(pinned)), "sha256": sha256(path),
                "source_url": "https://raw.githubusercontent.com/ggml-org/llama.cpp/"
                "d222767c7a6516559a3f49e7721b6c6b1acc87b4/" + str(path.relative_to(pinned))}
               for path in sorted(pinned.rglob("*")) if path.is_file()]
    compat = helper.parent / "ollama-compat"
    compat_sources = [{"name": path.name, "sha256": sha256(path),
                       "source_url": "https://raw.githubusercontent.com/ollama/ollama/"
                       "f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/llama/compat/" + path.name}
                      for path in sorted(compat.iterdir()) if path.is_file()]
    report = {
        "evidenceRole": "local_vocab_only_mechanism_probe_not_agent_benchmark",
        "actualModelCalls": 0, "contextCreated": False, "generationEnabled": False,
        "liveAdapterReady": False, "researchEvidenceEligible": False,
        "dirtyBinarySourceEquivalenceEstablished": False,
        "helper_path": str(helper), "helper_sha256": sha256(helper),
        "helper_source_sha256": sha256(Path(__file__).with_name("tokenizer.cpp")),
        "pinned_sources": sources,
        "compat_sources": compat_sources,
        "model_path": str(model), "model_bytes": model.stat().st_size,
        "model_digest_note": "Blob filename is an identifier; this probe does not rehash the full model.",
        "dependencies": dependencies, "fixtures": rows, "all_expected_decisions_correct": True,
        "limitations": ["No renderer or live generation binding is exercised.",
                        "Prompt-byte/token limits are not a CPU-time guarantee; the caller must enforce a process deadline.",
                        "Imports, pinned source control flow and observed skip log support vocab-only execution; "
                        "they do not establish equivalence of the installed dirty Ollama build to upstream source."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify(args.helper, args.model, args.output)
    print(json.dumps({"all_expected_decisions_correct": report["all_expected_decisions_correct"],
                      "fixtures": len(report["fixtures"]), "actualModelCalls": 0}))
