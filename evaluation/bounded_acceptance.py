"""Once-only R0 engineering acceptance; no live inference or source scripts.

Require reviewed, standalone materials; pin full declared dependencies before
the 24-arm schedule and recheck after it. This wrapper cannot approve research,
convert unknown semantic scores into passes, or enable a real-model endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from evaluation.bounded_controller import REGISTRY, run
from evaluation.bounded_freeze import freeze, local_spec, verify
from evaluation.bounded_pilot import checked_seal, make_protocol, seal, validate_references, write_new
from evaluation.bounded_preflight import Assets, file_digest
from evaluation.bounded_prepared_transport import broker_factory
from evaluation.bounded_reacceptance import claim, dependency_files, linked_counts, load_amendment, verify_parent
from evaluation.bounded_transport import strict_json
from network_runtime.contracts import sha256_json


MATERIAL_NAMES = ("cases.json", "references.json", "dialogues.json", "support.json")
ANNOTATIONS = {"a": "annotation-a.json", "b": "annotation-b.json"}
REQUIRED_EXTRAS = {"adjudication.json", "sources.json", "metadata.json"}


def _read_material(directory, name):
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("standalone regular material files required")
    raw = path.read_bytes()
    # Hash exactly the bytes parsed, not a second potentially different read.
    return strict_json(raw), "sha256:" + hashlib.sha256(raw).hexdigest()


def _material_names(review):
    return (*MATERIAL_NAMES, *ANNOTATIONS.values(), *sorted(review["extra_files"]), "review-manifest.json")


def load_material(directory):
    directory = Path(directory).resolve()
    review, _ = _read_material(directory, "review-manifest.json")
    review = checked_seal(review)
    annotations, extras = review.get("annotations"), review.get("extra_files")
    if (review.get("status") != "frozen_for_bounded_development"
            or review.get("role") != "AI_assisted_developer_evaluation_not_human_Gold"
            or review.get("unresolved_blockers") != []
            or not isinstance(annotations, dict) or set(annotations) != set(ANNOTATIONS)
            or not isinstance(extras, dict) or not REQUIRED_EXTRAS <= set(extras) or len(extras) > 32
            or any(not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", name) is None
                   or name in {*MATERIAL_NAMES, *ANNOTATIONS.values(), "review-manifest.json"} for name in extras)
            or not isinstance(review.get("files"), dict) or set(review["files"]) != set(MATERIAL_NAMES)):
        raise ValueError("two-role adjudicated material freeze required; drafts cannot run")
    expected = {**review["files"], **extras, **{name: annotations[role] for role, name in ANNOTATIONS.items()}}
    if any(not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
           for value in expected.values()):
        raise ValueError("complete raw SHA256 material bindings required")
    material = {}
    for name, digest in expected.items():
        value, actual = _read_material(directory, name)
        if actual != digest:
            raise ValueError("frozen material hash mismatch: " + name)
        if name in MATERIAL_NAMES:
            material[name] = value
    return material, review


def _bytecode_context():
    cache = os.environ.get("PYTHONPYCACHEPREFIX")
    if (not sys.dont_write_bytecode or not cache or not Path(cache).is_absolute()
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or Path(sys.pycache_prefix).resolve() != Path(cache).resolve()
            or (Path(cache).exists() and not Path(cache).is_dir())
            or (Path(cache).exists() and any(Path(cache).rglob("*")))):
        raise ValueError("use -B and an empty isolated absolute PYTHONPYCACHEPREFIX")
    return cache


def _controller_summary(report):
    """Counts describe execution, not assignment or business-semantic success."""
    counts = {"assigned_arms": 24, "claimed_arms": None, "protocol_completed_arms": None,
              "mechanics_passed_arms": None, "unrun_arms": None}
    if not isinstance(report, dict):
        return counts, False, {"type": None, "code": "controller_report_missing"}
    rows, observations, unrun = (report.get(name) for name in ("rows", "observations", "unrun"))
    if (not all(isinstance(value, list) for value in (rows, observations, unrun))
            or not all(isinstance(row, dict) for row in [*rows, *observations, *unrun])):
        return counts, False, {"type": None, "code": "controller_report_invalid"}
    counts.update(claimed_arms=len(rows),
        protocol_completed_arms=sum(row.get("protocol_completed") is True for row in observations),
        mechanics_passed_arms=sum(row.get("mechanics_passed") is True for row in rows), unrun_arms=len(unrun))
    completed = bool(report.get("controllerMechanicsPassed") is True and report.get("assigned_arms") == 24
                     and counts["claimed_arms"] == counts["protocol_completed_arms"] == counts["mechanics_passed_arms"] == 24
                     and len(observations) == 24 and counts["unrun_arms"] == 0
                     and all(row.get("error") is None for row in rows))
    if completed:
        return counts, True, None
    failed = next((row for row in rows if row.get("error") is not None or row.get("mechanics_passed") is not True), None)
    if failed is None:
        return counts, False, {"type": None, "code": "controller_batch_incomplete"}
    error = failed.get("error") if isinstance(failed.get("error"), dict) else {}

    def identifier(value):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None

    # Do not copy arbitrary exception messages, prompts or stderr into summary.
    return counts, False, {"type": identifier(error.get("type")), "code": "controller_arm_failed",
        "case_id": identifier(failed.get("case_id")), "arm": identifier(failed.get("arm")),
        "terminal": identifier(failed.get("terminal")), "stage": identifier(error.get("stage"))}


def acceptance(material_dir, assets_file, codec_file, output, *, reacceptance=None):
    output = Path(output).resolve()
    cache = _bytecode_context()
    if output.exists():
        raise ValueError("acceptance output must be new; fixed engineering ledger is not reset")
    material, review = load_material(material_dir)
    amendment = load_amendment(reacceptance, REGISTRY) if reacceptance is not None else None
    assets = Assets.load(assets_file)
    codec = {"path": str(Path(codec_file).resolve()), "sha256": file_digest(Path(codec_file).resolve())}
    trees, files, metadata = local_spec(assets=assets, codec=codec)
    # These standalone files are not under the project execution-code trees.
    # Include originals, addenda, adjudication, source text and the manifest,
    # not merely a declaration that two annotation hashes once existed.
    files.update({"material/" + name: Path(material_dir).resolve() / name for name in _material_names(review)})
    files["preflight/assets_manifest"] = Path(assets_file).resolve()
    if amendment is not None:
        files.update(dependency_files(reacceptance, amendment))
    metadata["bytecode_environment"] = {"dont_write_bytecode": True, "cache_prefix": cache}
    dependencies = freeze(trees, files, metadata=metadata)
    # Parsing and dependency hashing are distinct operations. Fail if files
    # changed between them; never pair old parsed labels with a new file hash.
    if load_material(material_dir) != (material, review):
        raise ValueError("material changed while preparing the dependency freeze")
    protocol = make_protocol("r0-reviewed-development-20260917", material["cases.json"],
        model_digest=assets.inspect()["model"]["sha256"],
        harness_digest=sha256_json(dependencies["trees"]["installed_dsh_node_modules"]),
        support=material["support.json"])
    references = validate_references(protocol, material["references.json"])
    bindings = {"protocol_digest": protocol["digest"], "review_digest": review["digest"],
                "dialogues_digest": sha256_json(material["dialogues.json"])}
    if amendment is not None and load_amendment(reacceptance, REGISTRY, bindings=bindings) != amendment:
        raise ValueError("reacceptance amendment changed during dependency freeze")
    output.mkdir(parents=True)
    write_new(output / "dependencies.json", dependencies)
    write_new(output / "protocol.json", protocol)
    write_new(output / "reference-freeze.json", review)
    write_new(output / "pre-run.json", seal({"dependency_digest": dependencies["digest"],
        "protocol_digest": protocol["digest"], "review_digest": review["digest"],
        "dialogues_digest": sha256_json(material["dialogues.json"]),
        "actualModelCalls": 0, "liveGenerationEnabled": False,
        **({"reacceptance_amendment_digest": amendment["digest"]} if amendment is not None else {})}))
    factory = broker_factory(assets, codec)
    report, failure, unchanged, cleanup, receipt = None, None, False, [], None
    try:
        verify(dependencies)
        if amendment is not None:
            if load_amendment(reacceptance, REGISTRY, bindings=bindings) != amendment:
                raise ValueError("reacceptance amendment drift before claim")
            receipt = claim(REGISTRY, amendment, output / "controller")
            write_new(output / "reacceptance-claim.json", receipt)
        report = run(protocol, references, output / "controller",
                     dialogues=material["dialogues.json"], broker_factory=factory,
                     **({"reacceptance_claim": receipt} if receipt is not None else {}))
        verify(dependencies)
        if receipt is not None:
            verify_parent(REGISTRY, receipt)
            if load_amendment(reacceptance, REGISTRY, bindings=bindings) != amendment:
                raise ValueError("reacceptance amendment drift after execution")
        if _bytecode_context() != cache:
            raise ValueError("isolated bytecode environment changed during acceptance")
        unchanged = True
    except Exception as exc:
        failure = {"type": type(exc).__name__, "code": "acceptance_failed_or_dependency_drift"}
    finally:
        # One failed cleanup must not skip the remaining receivers or hide the
        # controller/dependency failure and its full assigned denominator.
        for index, receiver in enumerate(factory.receivers):
            try:
                item = receiver.close()
                if (not isinstance(item, dict) or type(item.get("drained")) is not bool
                        or type(item.get("closed")) is not bool):
                    raise ValueError("receiver cleanup requires an explicit drain result")
                item = strict_json(json.dumps(item, allow_nan=False))
                cleanup.append({**item, "receiver_index": index})
            except Exception as exc:
                cleanup.append({"receiver_index": index, "closed": False, "drained": False,
                                "error_type": type(exc).__name__})
    counts, controller_complete, controller_failure = _controller_summary(report)
    cleanup_complete = all(item["closed"] and item["drained"] for item in cleanup)
    if failure is None:
        failure = controller_failure
    if failure is None and not cleanup_complete:
        failure = {"type": None, "code": "receiver_cleanup_failed"}
    if failure is None and len(cleanup) != counts["claimed_arms"]:
        failure = {"type": None, "code": "controller_receiver_count_mismatch"}
    linked = linked_counts(receipt, counts["claimed_arms"]) if receipt is not None else None
    if linked is not None:
        try:
            verify_parent(REGISTRY, receipt)
        except Exception:
            linked["parent_preserved"] = False
            unchanged = False
            failure = {"type": "ValueError", "code": "halted_parent_changed"}
    result = seal({"schema": "ensuredskill.io/r0-controller-acceptance/v2",
        "dependency_digest": dependencies["digest"], "dependencies_unchanged": unchanged,
        "review_digest": review["digest"], "protocol_digest": protocol["digest"],
        "controller_report_digest": report.get("digest") if isinstance(report, dict) else None,
        **counts,
        "created_receivers": len(cleanup),
        "closed_drained_receivers": sum(item["closed"] and item["drained"] for item in cleanup),
        "receiver_cleanup_status": "not_required" if not cleanup else "complete" if cleanup_complete else "failed",
        "mechanical_acceptance_passed": bool(controller_complete and unchanged and failure is None
                                              and len(cleanup) == 24 and cleanup_complete),
        "failure": failure, "receiver_cleanup": cleanup,
        "actualModelCalls": 0, "generationParity": "not_tested", "liveGenerationEnabled": False,
        "semanticPerformance": "not_measured", "researchEvidenceEligible": False,
        "r0Complete": False, "r0_completion_requires_separate_full_checklist": True,
        **({"reacceptance": linked} if linked is not None else {}),
        **({"reacceptance_amendment_digest": amendment["digest"]} if amendment is not None else {})})
    write_new(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("parser_codec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reacceptance", type=Path, help="explicit sealed one-time linked engineering amendment")
    args = parser.parse_args()
    result = acceptance(args.material, args.assets, args.parser_codec, args.output, reacceptance=args.reacceptance)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["mechanical_acceptance_passed"] else 1)


if __name__ == "__main__":
    main()
