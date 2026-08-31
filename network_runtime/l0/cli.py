"""Developer CLI for validating and inspecting L0 v2 contract packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .catalog import L0Catalog
from .compiler import L0CompileError
from .models import (
    AtomicEffectManifest,
    CompositeEffectManifest,
    DerivedEffectManifest,
)
from .promotion import (
    PromotionError,
    assess_promotion,
    build_l05_spec,
    inspect_skill,
    l05_yaml,
    package_promotion,
    promotion_prompt,
    review_promotion,
)
from .workbench import export_workbench_html, inspect_workbench, list_workbench


def _contract(catalog: L0Catalog, skill_id: str, version: str | None) -> Any:
    return catalog.require(skill_id, version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile, validate and explain NetOpYu L0 v2 Effect Contracts",
    )
    parser.add_argument(
        "--source", default="network_runtime/l0/examples",
        help="manifest file or directory (default: bundled S1/S11 examples)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="compile all manifests and verify references")
    sub.add_parser("list", help="list compiled contracts")

    for name in ("show", "explain", "graph"):
        command = sub.add_parser(name)
        command.add_argument("skill_id")
        command.add_argument("--version")

    diff_command = sub.add_parser(
        "diff", help="show a semantic review diff between two compiled contracts",
    )
    diff_command.add_argument("left_id")
    diff_command.add_argument("right_id")
    diff_command.add_argument("--left-version")
    diff_command.add_argument("--right-version")

    compile_command = sub.add_parser("compile", help="write the immutable compiled catalog")
    compile_command.add_argument("--output", required=True)
    schema_command = sub.add_parser("schema", help="print strict authoring JSON schemas")
    schema_command.add_argument(
        "--kind", choices=("atomic", "derived", "composite", "all"), default="all",
    )

    inspect_command = sub.add_parser(
        "promote-inspect", help="inspect one Anthropic-standard L1 SKILL.md",
    )
    inspect_command.add_argument("--skill", required=True)

    l05_command = sub.add_parser(
        "promote-l05", help="build the reviewable L0.5 structured-language stage",
    )
    l05_command.add_argument("--skill", required=True)
    l05_command.add_argument("--capabilities", required=True)
    l05_command.add_argument("--output", required=True)

    prompt_command = sub.add_parser(
        "promote-prompt", help="build a bounded Agent prompt packet for an L0 candidate",
    )
    prompt_command.add_argument("--skill", required=True)
    prompt_command.add_argument("--capabilities", required=True)
    prompt_command.add_argument("--l05")
    prompt_command.add_argument("--output")

    for name, help_text in (
        ("promote-assess", "cross-check and compile an untrusted L0 candidate"),
        ("promote-package", "create an immutable, non-activated review proposal"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--skill", required=True)
        command.add_argument("--candidate", required=True)
        command.add_argument("--capabilities", required=True)
        command.add_argument("--l05")
        command.add_argument(
            "--dependencies", action="append", default=[],
            help="L0 manifest file/directory required by a derived/composite candidate",
        )
        if name == "promote-package":
            command.add_argument("--output", required=True)

    review_command = sub.add_parser(
        "promote-review", help="record one human decision without activating Runtime",
    )
    review_command.add_argument("--proposal", required=True)
    review_command.add_argument("--reviewer", required=True)
    review_command.add_argument("--decision", choices=("approve", "reject"), required=True)
    review_command.add_argument("--reason", default="")

    workbench_inspect = sub.add_parser(
        "workbench-inspect",
        help="validate and project one immutable Promotion package for review",
    )
    workbench_inspect.add_argument("--proposal", required=True)
    workbench_list = sub.add_parser(
        "workbench-list",
        help="list direct-child Promotion packages without exposing directory names",
    )
    workbench_list.add_argument("--root", required=True)
    workbench_list.add_argument("--limit", type=int, default=100)
    workbench_export = sub.add_parser(
        "workbench-export",
        help="render a self-contained offline L0.5 review/editor HTML artifact",
    )
    workbench_export.add_argument("--proposal", required=True)
    workbench_export.add_argument("--output", required=True)

    core_evaluation = sub.add_parser(
        "core-eval-report",
        help="recompute the dual-core semantic-compilation and Runtime evidence report",
    )
    core_evaluation.add_argument(
        "--runtime-report", default="artifacts/runtime-ab/runtime-ab.json",
    )
    core_evaluation.add_argument(
        "--forward-report", default="artifacts/promotion-forward-calibration/report.json",
    )
    core_evaluation.add_argument(
        "--forward-model-report",
        default=(
            "artifacts/promotion-forward-model/"
            "qwen3.5-9b-p25c-v7-public-210/report.json"
        ),
    )
    core_evaluation.add_argument(
        "--runtime-reassessment-report",
        default=(
            "artifacts/promotion-forward-model/qwen3.5-9b-p25c-v7-public-210/"
            "current-runtime-reassessment/report.json"
        ),
    )
    core_evaluation.add_argument(
        "--json", default="artifacts/core-capability-evaluation/current.json",
    )
    core_evaluation.add_argument(
        "--markdown", default="docs/core-capability-evaluation-report.md",
    )

    forward_calibration = sub.add_parser(
        "forward-eval-calibrate",
        help="build the public 210-case reverse-bootstrap evaluator calibration matrix",
    )
    forward_calibration.add_argument(
        "--output-root", default="artifacts/promotion-forward-calibration",
    )
    forward_calibration.add_argument(
        "--markdown", default="docs/promotion-forward-qualification.md",
    )
    sub.add_parser(
        "forward-eval-schema",
        help="print Case, Label and Observation JSON Schemas for external qualification",
    )
    forward_inputs = sub.add_parser(
        "forward-eval-study-inputs",
        help="resolve private-study model/protocol/catalog digests without inference",
    )
    forward_inputs.add_argument("cases")
    forward_inputs.add_argument("--model", required=True)
    forward_inputs.add_argument("--base-url", default="http://127.0.0.1:11434")
    forward_inputs.add_argument("--timeout-seconds", type=float, default=30.0)
    forward_study = sub.add_parser(
        "forward-eval-study-init",
        help="pre-register immutable model inputs and separated private-study roles",
    )
    forward_study.add_argument("--dataset-id", required=True)
    forward_study.add_argument("--version", required=True)
    forward_study.add_argument("--case-author-id", action="append", required=True)
    forward_study.add_argument("--reviewer-id", action="append", required=True)
    forward_study.add_argument("--adjudicator-id", action="append", required=True)
    forward_study.add_argument("--model", required=True)
    forward_study.add_argument("--model-artifact-digest", required=True)
    forward_study.add_argument("--authoring-protocol-digest", required=True)
    forward_study.add_argument("--catalog-snapshot-digest", required=True)
    forward_study.add_argument("--repetitions", type=int, default=3)
    forward_study.add_argument("--output", required=True)
    forward_study_seal = sub.add_parser(
        "forward-eval-study-seal",
        help="seal private forward cases against a pre-registered study plan",
    )
    forward_study_seal.add_argument("cases")
    forward_study_seal.add_argument("study_plan")
    forward_study_seal.add_argument("--output", required=True)
    forward_review_packet = sub.add_parser(
        "forward-eval-review-pack",
        help="create one blinded reviewer packet without model outputs or gold labels",
    )
    forward_review_packet.add_argument("cases")
    forward_review_packet.add_argument("manifest")
    forward_review_packet.add_argument("study_plan")
    forward_review_packet.add_argument("--reviewer-id", required=True)
    forward_review_packet.add_argument("--output-root", required=True)
    forward_resolution_packet = sub.add_parser(
        "forward-eval-resolution-pack",
        help="create a private packet containing reviewer disagreements only",
    )
    forward_resolution_packet.add_argument("cases")
    forward_resolution_packet.add_argument("manifest")
    forward_resolution_packet.add_argument("study_plan")
    forward_resolution_packet.add_argument("reviewer_one")
    forward_resolution_packet.add_argument("reviewer_two")
    forward_resolution_packet.add_argument("--adjudicator-id", required=True)
    forward_resolution_packet.add_argument("--output-root", required=True)
    forward_seal = sub.add_parser(
        "forward-eval-seal",
        help="seal an external forward L1-to-L0 qualification case set",
    )
    forward_seal.add_argument("cases")
    forward_seal.add_argument("--dataset-id", required=True)
    forward_seal.add_argument("--version", required=True)
    forward_seal.add_argument(
        "--provenance",
        choices=("independent_forward", "reverse_bootstrap_calibration"),
        required=True,
    )
    forward_seal.add_argument("--output", required=True)
    forward_review = sub.add_parser(
        "forward-eval-adjudicate",
        help="compare two independent label sets against one sealed forward data set",
    )
    forward_review.add_argument("cases")
    forward_review.add_argument("manifest")
    forward_review.add_argument("reviewer_one")
    forward_review.add_argument("reviewer_two")
    forward_review.add_argument("--study-plan")
    forward_review.add_argument("--resolutions")
    forward_review.add_argument("--output", required=True)
    forward_score = sub.add_parser(
        "forward-eval-score",
        help="score repeated model observations without emitting prompts or labels",
    )
    forward_score.add_argument("cases")
    forward_score.add_argument("manifest")
    forward_score.add_argument("reviewer_one")
    forward_score.add_argument("reviewer_two")
    forward_score.add_argument("observations")
    forward_score.add_argument("--study-plan")
    forward_score.add_argument("--resolutions")
    forward_score.add_argument("--output", required=True)
    forward_record = sub.add_parser(
        "forward-eval-record",
        help="normalize one real agent result into a prompt-free qualification observation",
    )
    forward_record.add_argument("--case-id", required=True)
    forward_record.add_argument("--repetition", required=True, type=int)
    forward_record.add_argument("--model", required=True)
    forward_record.add_argument("--model-artifact-digest", required=True)
    forward_record.add_argument("--authoring-protocol-digest", required=True)
    forward_record.add_argument("--catalog-snapshot-digest", required=True)
    forward_record.add_argument(
        "--disposition",
        choices=("proposal", "clarify", "reject", "protocol_error"),
        required=True,
    )
    forward_record.add_argument("--proposal")
    forward_record.add_argument("--catalog-id")
    forward_record.add_argument("--missing-field", action="append", default=[])
    forward_record.add_argument("--latency-ms", required=True, type=float)
    forward_record.add_argument("--model-calls", required=True, type=int)
    forward_record.add_argument("--repair-attempts", type=int, default=0)
    forward_record.add_argument("--input-tokens", type=int, default=0)
    forward_record.add_argument("--output-tokens", type=int, default=0)
    forward_model = sub.add_parser(
        "forward-eval-run-model",
        help="run a local Ollama model through real L1-to-L0 Promotion checks",
    )
    forward_model.add_argument("--model", default="qwen3.5:9b")
    forward_model.add_argument("--base-url", default="http://127.0.0.1:11434")
    forward_model.add_argument(
        "--output-root", default="artifacts/promotion-forward-model/qwen3.5-9b",
    )
    forward_model.add_argument("--limit", type=int)
    forward_model.add_argument(
        "--family", action="append", default=[],
        help="run only one public calibration family; may be repeated",
    )
    forward_model.add_argument(
        "--case-id", action="append", default=[],
        help="run only one public calibration case id; may be repeated",
    )
    forward_model.add_argument("--repetitions", type=int, default=1)
    forward_model.add_argument("--timeout-seconds", type=float, default=180.0)
    forward_model.add_argument("--repair-limit", type=int, default=1)
    forward_model.add_argument(
        "--transport-failure-limit", type=int, default=2,
        help=(
            "pause after this many consecutive model transport failures; "
            "0 disables the circuit breaker"
        ),
    )
    forward_model.add_argument(
        "--resume", action="store_true",
        help="resume the fingerprint-bound active run from per-case checkpoints",
    )
    forward_private = sub.add_parser(
        "forward-eval-run-private",
        help="run one pre-registered private study through the real model/Promotion path",
    )
    forward_private.add_argument("cases")
    forward_private.add_argument("manifest")
    forward_private.add_argument("study_plan")
    forward_private.add_argument("reviewer_one")
    forward_private.add_argument("reviewer_two")
    forward_private.add_argument("--resolutions")
    forward_private.add_argument("--model", required=True)
    forward_private.add_argument("--base-url", default="http://127.0.0.1:11434")
    forward_private.add_argument("--output-root", required=True)
    forward_private.add_argument("--repetitions", type=int, default=3)
    forward_private.add_argument("--timeout-seconds", type=float, default=180.0)
    forward_private.add_argument("--repair-limit", type=int, default=1)
    forward_private.add_argument(
        "--transport-failure-limit", type=int, default=2,
        help=(
            "pause after this many consecutive model transport failures; "
            "0 disables the circuit breaker"
        ),
    )
    forward_private.add_argument("--resume", action="store_true")
    forward_rescore = sub.add_parser(
        "forward-eval-rescore-model",
        help="re-run deterministic scoring over an existing model run without inference",
    )
    forward_rescore.add_argument(
        "--output-root", default="artifacts/promotion-forward-model/qwen3.5-9b",
    )
    forward_reassess = sub.add_parser(
        "forward-eval-reassess-runtime",
        help=(
            "replay stored semantic proposals through the current Runtime without "
            "calling the model"
        ),
    )
    forward_reassess.add_argument(
        "--output-root", default="artifacts/promotion-forward-model/qwen3.5-9b",
    )

    sub.add_parser(
        "runtime-validate", help="validate the complete activated production L0 v2 catalog",
    )
    sub.add_parser(
        "runtime-list", help="list every activated production L0 v2 contract",
    )
    runtime_export = sub.add_parser(
        "runtime-export", help="export the immutable activated production catalog",
    )
    runtime_export.add_argument("--output", required=True)
    trajectory_build = sub.add_parser(
        "runtime-trajectories-build",
        help="materialize readable L1/L0.5/L0 archives for every production L0",
    )
    trajectory_build.add_argument(
        "--output", default="network_runtime/l0/production_trajectories",
    )
    trajectory_validate = sub.add_parser(
        "runtime-trajectories-validate",
        help="validate stage hashes, Promotion parity, and exact L0 round trips",
    )
    trajectory_validate.add_argument(
        "--source", default="network_runtime/l0/production_trajectories",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            models = {
                "atomic": AtomicEffectManifest,
                "derived": DerivedEffectManifest,
                "composite": CompositeEffectManifest,
            }
            selected = models if args.kind == "all" else {args.kind: models[args.kind]}
            print(json.dumps(
                {name: model.model_json_schema(by_alias=True) for name, model in selected.items()},
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
            return 0
        if args.command == "promote-inspect":
            print(json.dumps(inspect_skill(args.skill), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "promote-l05":
            value = l05_yaml(build_l05_spec(
                skill_path=args.skill,
                capability_catalog_path=args.capabilities,
            ))
            destination = Path(args.output).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(value, encoding="utf-8")
            print(json.dumps({"ok": True, "output": str(destination)}, ensure_ascii=False))
            return 0
        if args.command == "promote-prompt":
            value = promotion_prompt(
                skill_path=args.skill, capability_catalog_path=args.capabilities,
                l05_path=args.l05,
            )
            if args.output:
                destination = Path(args.output).expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(value, encoding="utf-8")
                print(json.dumps({"ok": True, "output": str(destination)}, ensure_ascii=False))
            else:
                print(value, end="")
            return 0
        if args.command == "promote-assess":
            assessment = assess_promotion(
                skill_path=args.skill, candidate_path=args.candidate,
                capability_catalog_path=args.capabilities,
                dependency_paths=args.dependencies,
                l05_path=args.l05,
            )
            print(json.dumps(assessment.report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if assessment.report["status"] == "ready_for_review" else 1
        if args.command == "promote-package":
            print(json.dumps(package_promotion(
                skill_path=args.skill, candidate_path=args.candidate,
                capability_catalog_path=args.capabilities,
                dependency_paths=args.dependencies, output_directory=args.output,
                l05_path=args.l05,
            ), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "promote-review":
            print(json.dumps(review_promotion(
                proposal_directory=args.proposal, reviewer=args.reviewer,
                decision=args.decision, reason=args.reason,
            ), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "workbench-inspect":
            print(json.dumps(
                inspect_workbench(args.proposal),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
            return 0
        if args.command == "workbench-list":
            print(json.dumps(
                list_workbench(args.root, limit=args.limit),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
            return 0
        if args.command == "workbench-export":
            print(json.dumps(
                export_workbench_html(args.proposal, args.output),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
            return 0
        if args.command == "core-eval-report":
            from network_runtime.l0.core_capability_evaluation import (
                write_core_capability_evaluation,
            )

            print(json.dumps(
                write_core_capability_evaluation(
                    runtime_report_path=args.runtime_report,
                    forward_report_path=args.forward_report,
                    forward_model_report_path=args.forward_model_report,
                    runtime_reassessment_report_path=(
                        args.runtime_reassessment_report
                    ),
                    json_path=args.json,
                    markdown_path=args.markdown,
                ),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
            return 0
        if args.command.startswith("forward-eval-"):
            from network_runtime.l0.forward_qualification import (
                adjudicate_forward_labels,
                build_forward_resolution_packet,
                build_forward_review_packet,
                create_forward_study_plan,
                forward_qualification_schemas,
                qualify_forward_files,
                record_forward_observation,
                seal_forward_cases,
                seal_forward_study,
                write_public_calibration,
            )

            if args.command in {
                "forward-eval-run-model", "forward-eval-run-private",
                "forward-eval-rescore-model",
                "forward-eval-reassess-runtime", "forward-eval-study-inputs",
            }:
                from network_runtime.l0.forward_model_runner import (
                    inspect_private_study_inputs,
                    reassess_public_model_evaluation,
                    rescore_public_model_evaluation,
                    run_public_model_evaluation,
                )

                if args.command == "forward-eval-study-inputs":
                    value = inspect_private_study_inputs(
                        args.cases, model=args.model, base_url=args.base_url,
                        timeout_seconds=args.timeout_seconds,
                    )
                elif args.command in {
                    "forward-eval-run-model", "forward-eval-run-private",
                }:
                    value = run_public_model_evaluation(
                        model=args.model,
                        base_url=args.base_url,
                        output_root=args.output_root,
                        limit=(args.limit if args.command == "forward-eval-run-model" else None),
                        families=(
                            tuple(args.family)
                            if args.command == "forward-eval-run-model" else ()
                        ),
                        case_ids=(
                            tuple(args.case_id)
                            if args.command == "forward-eval-run-model" else ()
                        ),
                        repetitions=args.repetitions,
                        timeout_seconds=args.timeout_seconds,
                        repair_limit=args.repair_limit,
                        transport_failure_limit=args.transport_failure_limit,
                        resume=args.resume,
                        private_cases_path=(
                            args.cases if args.command == "forward-eval-run-private" else None
                        ),
                        private_manifest_path=(
                            args.manifest if args.command == "forward-eval-run-private" else None
                        ),
                        private_study_plan_path=(
                            args.study_plan if args.command == "forward-eval-run-private" else None
                        ),
                        private_reviewer_one_path=(
                            args.reviewer_one if args.command == "forward-eval-run-private" else None
                        ),
                        private_reviewer_two_path=(
                            args.reviewer_two if args.command == "forward-eval-run-private" else None
                        ),
                        private_resolutions_path=(
                            args.resolutions if args.command == "forward-eval-run-private" else None
                        ),
                    )
                elif args.command == "forward-eval-rescore-model":
                    value = rescore_public_model_evaluation(args.output_root)
                else:
                    value = reassess_public_model_evaluation(args.output_root)
            elif args.command == "forward-eval-schema":
                value = forward_qualification_schemas()
            elif args.command == "forward-eval-calibrate":
                value = write_public_calibration(
                    output_root=args.output_root,
                    markdown_path=args.markdown,
                )
            elif args.command == "forward-eval-study-init":
                value = create_forward_study_plan(
                    dataset_id=args.dataset_id,
                    version=args.version,
                    case_author_ids=args.case_author_id,
                    reviewer_ids=args.reviewer_id,
                    adjudicator_ids=args.adjudicator_id,
                    model=args.model,
                    model_artifact_digest=args.model_artifact_digest,
                    authoring_protocol_digest=args.authoring_protocol_digest,
                    catalog_snapshot_digest=args.catalog_snapshot_digest,
                    repetitions=args.repetitions,
                )
            elif args.command == "forward-eval-study-seal":
                value = seal_forward_study(args.cases, args.study_plan)
            elif args.command == "forward-eval-review-pack":
                value = build_forward_review_packet(
                    args.cases, args.manifest, args.study_plan,
                    reviewer_id=args.reviewer_id, output_root=args.output_root,
                )
            elif args.command == "forward-eval-resolution-pack":
                value = build_forward_resolution_packet(
                    args.cases, args.manifest, args.study_plan,
                    args.reviewer_one, args.reviewer_two,
                    adjudicator_id=args.adjudicator_id,
                    output_root=args.output_root,
                )
            elif args.command == "forward-eval-seal":
                value = seal_forward_cases(
                    args.cases,
                    dataset_id=args.dataset_id,
                    version=args.version,
                    provenance=args.provenance,
                )
            elif args.command == "forward-eval-adjudicate":
                value = adjudicate_forward_labels(
                    args.cases, args.manifest, args.reviewer_one, args.reviewer_two,
                    study_plan_path=args.study_plan,
                    resolutions_path=args.resolutions,
                )
            elif args.command == "forward-eval-score":
                value = qualify_forward_files(
                    args.cases, args.manifest, args.reviewer_one,
                    args.reviewer_two, args.observations,
                    study_plan_path=args.study_plan,
                    resolutions_path=args.resolutions,
                )
            else:
                value = record_forward_observation(
                    case_id=args.case_id,
                    repetition=args.repetition,
                    model=args.model,
                    model_artifact_digest=args.model_artifact_digest,
                    authoring_protocol_digest=args.authoring_protocol_digest,
                    catalog_snapshot_digest=args.catalog_snapshot_digest,
                    disposition=args.disposition,
                    latency_ms=args.latency_ms,
                    model_calls=args.model_calls,
                    repair_attempts=args.repair_attempts,
                    input_tokens=args.input_tokens,
                    output_tokens=args.output_tokens,
                    proposal_path=args.proposal,
                    catalog_id=args.catalog_id,
                    missing_fields=args.missing_field,
                ).model_dump(by_alias=True, mode="json")
            if hasattr(args, "output"):
                destination = Path(args.output).expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            if args.command == "forward-eval-score" and not value["qualified"]:
                return 1
            return 0
        if args.command in {
            "runtime-trajectories-build", "runtime-trajectories-validate",
        }:
            from network_runtime.l0.production_trajectory import (
                build_production_trajectories,
                validate_production_trajectories,
            )

            value = (
                build_production_trajectories(args.output)
                if args.command == "runtime-trajectories-build"
                else validate_production_trajectories(args.source)
            )
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command in {"runtime-validate", "runtime-list", "runtime-export"}:
            from network_runtime.l0.production import BINDINGS, CATALOG
            from network_runtime.l0_skills import REGISTRY as RUNTIME_REGISTRY
            from network_runtime.policies import reviewed_contracts

            contracts = CATALOG.contracts()
            if args.command == "runtime-validate":
                from network_runtime.l0.production_trajectory import (
                    validate_production_trajectories,
                )

                runtime_tools = {item.tool_name for item in RUNTIME_REGISTRY.contracts()}
                reviewed_tools = set(reviewed_contracts())
                if runtime_tools != reviewed_tools or len(BINDINGS) != len(contracts):
                    raise L0CompileError("production v2 catalog coverage mismatch")
                trajectories = validate_production_trajectories()
                print(json.dumps({
                    "ok": True,
                    "contracts": len(contracts),
                    "bindings": len(BINDINGS),
                    "reviewed_write_tools": len(reviewed_tools),
                    "readable_trajectories": trajectories["contracts"],
                    "promotion_ready": trajectories["promotion_ready"],
                    "exact_round_trips": trajectories["exact_round_trips"],
                    "formats": sorted({item.api_version for item in contracts}),
                    "runtime_authority": "l0-v2-compiled",
                }, ensure_ascii=False, indent=2, sort_keys=True))
            elif args.command == "runtime-list":
                for item in contracts:
                    binding = BINDINGS[(item.metadata.id, item.metadata.version)]
                    print(
                        f"{item.metadata.id}@{item.metadata.version}\t{binding.tool_name}\t"
                        f"{item.contract_hash}"
                    )
            else:
                destination = Path(args.output).expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(CATALOG.to_json(), encoding="utf-8")
                print(json.dumps({
                    "ok": True, "contracts": len(contracts), "output": str(destination),
                }, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        catalog = L0Catalog.from_path(args.source)
        if args.command == "validate":
            atomic = sum(item.kind == "CompiledAtomicEffect" for item in catalog.contracts())
            composite = len(catalog.contracts()) - atomic
            print(json.dumps({
                "ok": True,
                "source": str(Path(args.source).expanduser().resolve()),
                "contracts": len(catalog.contracts()),
                "atomic_or_derived": atomic,
                "composite": composite,
            }, ensure_ascii=False, indent=2))
        elif args.command == "list":
            for item in catalog.contracts():
                relation = getattr(item, "derivation", "composite")
                print(f"{item.metadata.id}@{item.metadata.version}\t{relation}\t{item.metadata.description}")
        elif args.command == "show":
            value = _contract(catalog, args.skill_id, args.version)
            print(json.dumps(
                value.model_dump(by_alias=True, mode="json"),
                ensure_ascii=False, indent=2, sort_keys=True,
            ))
        elif args.command == "explain":
            print(catalog.explain(args.skill_id, args.version))
        elif args.command == "graph":
            print(catalog.graph(args.skill_id, args.version))
        elif args.command == "diff":
            print(json.dumps(catalog.diff(
                args.left_id, args.right_id,
                args.left_version, args.right_version,
            ), ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "compile":
            destination = Path(args.output).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(catalog.to_json(), encoding="utf-8")
            print(json.dumps({
                "ok": True, "contracts": len(catalog.contracts()),
                "output": str(destination),
            }, ensure_ascii=False, indent=2))
        return 0
    except (L0CompileError, PromotionError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({
            "ok": False, "error": f"{type(error).__name__}: {error}",
        }, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
