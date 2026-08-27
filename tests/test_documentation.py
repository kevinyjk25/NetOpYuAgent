"""Contract tests for the canonical bilingual project documentation."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCUMENTS = (
    "README.md",
    "ARCHITECTURE.md",
    "HLD.md",
    "LLD.md",
    "SSD.md",
)
RETIRED_DOCUMENTS = (
    "DSH_MIGRATION.md",
    "L1_L0_SKILL_DEMO.md",
    "MIGRATION_AUDIT.md",
    "NETWORK_RUNTIME.md",
)


def test_canonical_documents_are_bilingual_with_chinese_first() -> None:
    for name in CANONICAL_DOCUMENTS:
        content = (ROOT / name).read_text(encoding="utf-8")
        chinese = content.index("## 中文")
        english = content.index("## English")
        assert chinese < english, f"{name} must place Chinese before English"


def test_local_document_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+\.md)\)")
    for name in CANONICAL_DOCUMENTS:
        content = (ROOT / name).read_text(encoding="utf-8")
        for target in link_pattern.findall(content):
            assert (ROOT / target).is_file(), f"{name} links to missing {target}"


def test_duplicate_migration_documents_stay_retired() -> None:
    assert all(not (ROOT / name).exists() for name in RETIRED_DOCUMENTS)
