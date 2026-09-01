"""Domain-neutral public façade for L1 -> L0.5 -> L0 Promotion.

The implementation remains in ``network_runtime.l0`` during the compatibility
period so existing network profiles and stored artifact paths do not break.
New integrations should import this module; network is the first reference
profile, not the ownership boundary of the compiler.
"""

from network_runtime.l0.promotion import (
    CapabilityCatalogManifest,
    CapabilityDefinition,
    PromotionAssessment,
    PromotionError,
    StructuredNaturalLanguageSkill,
    assess_promotion,
    build_l05_spec,
    inspect_skill,
    l05_yaml,
    package_promotion,
    promotion_prompt,
    review_promotion,
)

__all__ = [
    "CapabilityCatalogManifest", "CapabilityDefinition", "PromotionAssessment",
    "PromotionError", "StructuredNaturalLanguageSkill", "assess_promotion",
    "build_l05_spec", "inspect_skill", "l05_yaml", "package_promotion",
    "promotion_prompt", "review_promotion",
]
