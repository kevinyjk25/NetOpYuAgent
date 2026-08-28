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
    inspect_skill,
    package_promotion,
    promotion_prompt,
    review_promotion,
)


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

    prompt_command = sub.add_parser(
        "promote-prompt", help="build a bounded Agent prompt packet for an L0 candidate",
    )
    prompt_command.add_argument("--skill", required=True)
    prompt_command.add_argument("--capabilities", required=True)
    prompt_command.add_argument("--output")

    for name, help_text in (
        ("promote-assess", "cross-check and compile an untrusted L0 candidate"),
        ("promote-package", "create an immutable, non-activated review proposal"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--skill", required=True)
        command.add_argument("--candidate", required=True)
        command.add_argument("--capabilities", required=True)
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
        if args.command == "promote-prompt":
            value = promotion_prompt(
                skill_path=args.skill, capability_catalog_path=args.capabilities,
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
            )
            print(json.dumps(assessment.report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if assessment.report["status"] == "ready_for_review" else 1
        if args.command == "promote-package":
            print(json.dumps(package_promotion(
                skill_path=args.skill, candidate_path=args.candidate,
                capability_catalog_path=args.capabilities,
                dependency_paths=args.dependencies, output_directory=args.output,
            ), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "promote-review":
            print(json.dumps(review_promotion(
                proposal_directory=args.proposal, reviewer=args.reviewer,
                decision=args.decision, reason=args.reason,
            ), ensure_ascii=False, indent=2, sort_keys=True))
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
