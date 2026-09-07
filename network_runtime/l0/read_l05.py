"""Readable single-read L0.5 proposals, separate from effect transaction semantics."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .models import API_VERSION, AtomicReadManifest, AtomicReadSpec, Metadata, StrictModel


class ReadL05Proposal(StrictModel):
    api_version: Literal["netopyu.io/l0.5-read-proposal/v1"] = Field(alias="apiVersion")
    kind: Literal["ReadL05Proposal"]
    metadata: Metadata
    purpose: str = Field(min_length=1, max_length=4000)
    scope: Literal["single_read_operation"]
    operation: AtomicReadSpec
    unresolved_questions: tuple[str, ...] = Field(alias="unresolvedQuestions")

    def to_manifest(self) -> AtomicReadManifest:
        # Revalidate mutable nested values. Purpose must not disappear during lowering.
        proposal = ReadL05Proposal.model_validate(self.model_dump(by_alias=True))
        if proposal.metadata.description != proposal.purpose:
            raise ValueError("L0.5 purpose must equal metadata.description; resolve conflicting descriptions")
        return AtomicReadManifest(
            apiVersion=API_VERSION, kind="AtomicRead", metadata=proposal.metadata,
            spec=proposal.operation,
        )


def scaffold_read_l05(manifest: AtomicReadManifest) -> ReadL05Proposal:
    """Mechanical reverse scaffold, explicitly NOT an L1 translation experiment."""
    manifest = AtomicReadManifest.model_validate(manifest.model_dump(by_alias=True))
    return ReadL05Proposal(
        apiVersion="netopyu.io/l0.5-read-proposal/v1", kind="ReadL05Proposal",
        metadata=manifest.metadata, purpose=manifest.metadata.description,
        scope="single_read_operation", operation=manifest.spec,
        unresolvedQuestions=("Review source fidelity before using this reverse-generated scaffold.",),
    )
