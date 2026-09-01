"""Contract tests for the canonical bilingual project documentation."""

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCUMENTS = (
    "README.md",
    "ARCHITECTURE.md",
    "HLD.md",
    "LLD.md",
    "SSD.md",
)
PROJECT_DOCUMENTS = tuple(
    sorted(
        [ROOT / name for name in CANONICAL_DOCUMENTS]
        + list((ROOT / "docs").rglob("*.md"))
        + list((ROOT / "labs").glob("*/README.md"))
        + [ROOT / "network_runtime/l0/production_trajectories/INDEX.md"]
    )
)
RETIRED_DOCUMENTS = (
    "DSH_MIGRATION.md",
    "L1_L0_SKILL_DEMO.md",
    "MIGRATION_AUDIT.md",
    "NETWORK_RUNTIME.md",
)


def test_canonical_documents_are_bilingual_with_chinese_first() -> None:
    for path in PROJECT_DOCUMENTS:
        # Submission drafts are deliberately maintained as structurally paired
        # Chinese and English files so either can be edited and anonymized
        # independently.  The documentation map declares this sole exception.
        if path.parent.name == "research" and "Paper_Draft" in path.name:
            continue
        content = path.read_text(encoding="utf-8")
        chinese = content.index("## 中文")
        english = content.index("## English")
        assert chinese < english, f"{path.relative_to(ROOT)} must place Chinese first"


def test_local_document_links_resolve() -> None:
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for source in PROJECT_DOCUMENTS:
        content = source.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(content):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not path_text:
                continue
            resolved = (source.parent / path_text).resolve()
            assert resolved.exists(), (
                f"{source.relative_to(ROOT)} links to missing {target}"
            )


def test_duplicate_migration_documents_stay_retired() -> None:
    assert all(not (ROOT / name).exists() for name in RETIRED_DOCUMENTS)
