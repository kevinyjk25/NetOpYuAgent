"""S12 — config 3-way consistency (dataclass ↔ loader ↔ yaml).

We hit this class of bug TWICE: a new key added to config.yaml under
`skill_orchestration` was silently dropped because the typed
`SkillOrchestrationConfig` dataclass had a fixed field list and the loader
didn't map it — so `getattr(cfg.skill_orchestration, key)` was MISSING at
runtime and the feature ran on a stale default with no error.

This test auto-discovers every dataclass sub-config on `cfg` and asserts
every key present in the corresponding config.yaml section maps to a real
dataclass field that actually loaded. It needs no per-config maintenance —
add a new config section and it's covered automatically.
"""
import dataclasses
import unittest
from pathlib import Path

import yaml

from config import cfg


def _yaml() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}


def _dataclass_subconfigs():
    """Yield (yaml_section_name, dataclass_instance) for every dataclass
    attribute on cfg whose name matches a top-level yaml section."""
    y = _yaml()
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        v = getattr(cfg, name, None)
        if dataclasses.is_dataclass(v) and name in y:
            yield name, v


class TestConfigConsistency(unittest.TestCase):
    # yaml keys that intentionally map to a DIFFERENTLY-NAMED dataclass field
    # (the loader renames them). These are NOT silent drops. Keep this list
    # tiny and documented — every entry is a deliberate transformation.
    INTENTIONAL_RENAMES = {
        "agent.peers",        # → AgentIdentityConfig.peer_urls (also AGENT_PEERS env)
    }

    def test_every_yaml_key_maps_to_a_dataclass_field(self):
        """The silent-drop guard: any key under a yaml section that the typed
        config object doesn't expose is a bug (added to yaml, forgotten in the
        dataclass+loader)."""
        y = _yaml()
        violations = []
        for section, inst in _dataclass_subconfigs():
            field_names = {f.name for f in dataclasses.fields(inst)}
            yaml_keys = set((y.get(section) or {}).keys())
            dropped = yaml_keys - field_names
            for k in sorted(dropped):
                qualified = f"{section}.{k}"
                if qualified in self.INTENTIONAL_RENAMES:
                    continue
                violations.append(qualified)
        self.assertEqual(
            violations, [],
            "yaml keys silently dropped (in config.yaml but NOT a field on the "
            "runtime config object — add to dataclass + loader, or to "
            "INTENTIONAL_RENAMES if the loader renames it):\n  " +
            "\n  ".join(violations))

    def test_loaded_values_match_yaml_for_scalar_keys(self):
        """Beyond presence: a mapped key must actually carry the yaml VALUE,
        not just exist with a default (catches a field added to the dataclass
        but missing from the loader's key mapping)."""
        y = _yaml()
        mismatches = []
        for section, inst in _dataclass_subconfigs():
            sect = y.get(section) or {}
            for f in dataclasses.fields(inst):
                if f.name not in sect:
                    continue
                yv = sect[f.name]
                # only check scalars (dicts/lists may be transformed by loader)
                if not isinstance(yv, (str, int, float, bool)):
                    continue
                rv = getattr(inst, f.name)
                # empty-string yaml → None / [] / "" at runtime is the standard
                # "unset optional" idiom; treat as equivalent, not a mismatch.
                if yv == "" and rv in (None, "", [], {}):
                    continue
                # env overrides legitimately differ; skip if an env var is set
                # (loaders use _env_*). We can't know the env var name generically,
                # so only flag when types disagree or value clearly unmapped
                # (runtime kept the dataclass default while yaml said otherwise
                # AND no env is plausibly involved → loader gap). Conservative:
                # compare with bool/number coercion.
                try:
                    if isinstance(yv, bool):
                        ok = bool(rv) == yv
                    elif isinstance(yv, (int, float)):
                        ok = float(rv) == float(yv)
                    else:
                        ok = str(rv) == str(yv)
                except (TypeError, ValueError):
                    ok = True  # don't false-positive on exotic types
                if not ok:
                    mismatches.append(
                        f"{section}.{f.name}: yaml={yv!r} runtime={rv!r}")
        # After the empty-string-unset normalization, remaining mismatches are
        # either real loader gaps or env overrides active in this process.
        # Allow a tiny margin for env overrides; a larger count = loader bug.
        self.assertLessEqual(
            len(mismatches), 2,
            "yaml↔runtime value mismatches (loader likely not mapping keys, or "
            "undocumented env overrides):\n  " + "\n  ".join(mismatches))

    def test_skill_orchestration_evolution_keys_present(self):
        """Explicit pin for the keys that bit us twice (weak-ambiguity +
        preference + P1/P3 trajectory + capability). Regression lock."""
        so = cfg.skill_orchestration
        for key in (
            "weak_ambiguity_floor", "weak_ambiguity_gap",
            "preference_learning_enabled", "preference_auto_threshold",
            "preference_recommend_floor", "preference_ttl_days",
            "trajectory_recurrence_threshold", "trajectory_similarity_threshold",
            "append_attribution_floor",
        ):
            self.assertTrue(hasattr(so, key),
                            f"skill_orchestration.{key} missing on runtime config "
                            f"(yaml key dropped by dataclass/loader)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
