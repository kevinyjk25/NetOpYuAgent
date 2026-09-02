"""L0 v2 authoring SDK: strict manifests, compiler and multi-version catalog.

The existing :mod:`network_runtime.l0_skills` module remains the production
compatibility registry while v2 is qualified.  V2 authoring definitions are
compiled and flattened before a Runtime can consume them; inheritance is never
interpreted dynamically during an approved execution.
"""

from .catalog import L0Catalog
from .compiler import L0CompileError, compile_documents, load_documents
from .models import (
    AtomicEffectManifest,
    CompiledAtomicEffect,
    CompiledCompositeEffect,
    CompositeEffectManifest,
    DerivedEffectManifest,
    SkillRef,
)
from .promotion import assess_promotion, package_promotion, promotion_prompt
from .workbench import export_workbench_html, inspect_workbench, list_workbench

__all__ = [
    "AtomicEffectManifest",
    "CompiledAtomicEffect",
    "CompiledCompositeEffect",
    "CompositeEffectManifest",
    "DerivedEffectManifest",
    "L0Catalog",
    "L0CompileError",
    "SkillRef",
    "compile_documents",
    "load_documents",
    "assess_promotion",
    "package_promotion",
    "promotion_prompt",
    "export_workbench_html",
    "inspect_workbench",
    "list_workbench",
]
