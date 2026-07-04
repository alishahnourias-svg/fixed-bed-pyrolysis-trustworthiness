#!/usr/bin/env python3
"""
Correct the live repository metadata after the manuscript title was finalized.

Run from the repository root:
    python tools/fix_live_repository_metadata.py

Changes:
- README.md: replace the old manuscript title with the final title.
- README.md: replace the obsolete pending-DOI wording with the issued DOI.
- README.md: update release status to reflect Zenodo archival.
- CITATION.cff: replace the old title, set version 1.0.0, and record the
  issued DOI and GitHub repository URL.
- release_metadata/PACKAGE_AUDIT.json: update DOI/title fields when present.
- FILE_INVENTORY.tsv and MANIFEST_SHA256.tsv: regenerate after changes.
- verify that the obsolete title and pending-DOI sentence no longer remain
  in README.md or CITATION.cff.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

OLD_TITLE = (
    "From Accuracy to Trustworthiness: Grouped Validation, "
    "Applicability-Domain Mapping, and Reliability-Aware Optimization "
    "for Machine-Learning-Based Fixed-Bed Biomass Pyrolysis Yield Prediction"
)

NEW_TITLE = (
    "From Accuracy to Trustworthiness: Grouped Validation, "
    "Applicability-Domain Mapping, and Reliability-Aware Scenario Screening "
    "for Machine-Learning-Based Fixed-Bed Biomass Pyrolysis Yield Prediction"
)

OLD_DOI_SENTENCE = (
    "The version-specific public software DOI will be minted by Zenodo after "
    "GitHub release `v1.0.0` is published."
)

NEW_DOI_SENTENCE = (
    "The version-specific public software release is archived in Zenodo under "
    "DOI 10.5281/zenodo.21176727."
)

OLD_STATUS = (
    "**Status:** Public software release `1.0.0`, prepared for DOI-bearing "
    "archival through the Zenodo–GitHub integration."
)

NEW_STATUS = (
    "**Status:** Public software release `1.0.0`, archived in Zenodo under "
    "DOI `10.5281/zenodo.21176727`."
)

PUBLIC_DOI = "10.5281/zenodo.21176727"
GITHUB_URL = (
    "https://github.com/alishahnourias-svg/"
    "fixed-bed-pyrolysis-trustworthiness"
)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run this script from the repository root."
        )


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(
            f"Expected text was not found while updating {label}:\n{old}"
        )
    return text.replace(old, new)


def classify(rel: str) -> str:
    if rel.startswith(".github/"):
        return "github_automation"
    if rel.startswith("code/") or rel.startswith("tools/"):
        return "code_or_tool"
    if rel.startswith("figures/"):
        return "figure"
    if rel.startswith("results/"):
        return "aggregate_result"
    if rel.startswith("release_metadata/"):
        return "release_metadata"
    return "root_metadata"


def main() -> int:
    root = Path.cwd().resolve()
    readme_path = root / "README.md"
    citation_path = root / "CITATION.cff"
    inventory_path = root / "FILE_INVENTORY.tsv"
    manifest_path = root / "MANIFEST_SHA256.tsv"

    for path in (readme_path, citation_path, inventory_path, manifest_path):
        require(path)

    # README.md
    readme = readme_path.read_text(encoding="utf-8")
    readme = replace_required(readme, OLD_TITLE, NEW_TITLE, "README title")
    readme = replace_required(
        readme, OLD_DOI_SENTENCE, NEW_DOI_SENTENCE, "README DOI sentence"
    )
    if OLD_STATUS in readme:
        readme = readme.replace(OLD_STATUS, NEW_STATUS)
    readme_path.write_text(readme, encoding="utf-8", newline="\n")

    # CITATION.cff
    citation = citation_path.read_text(encoding="utf-8")
    citation = replace_required(
        citation, OLD_TITLE, NEW_TITLE, "CITATION.cff title"
    )
    citation = citation.replace("version: 1.0.0-rc4", "version: 1.0.0")

    if "doi:" not in citation:
        anchor = "license: MIT\n"
        if anchor not in citation:
            raise RuntimeError(
                "Could not find 'license: MIT' in CITATION.cff to insert DOI."
            )
        citation = citation.replace(
            anchor,
            f"license: MIT\ndoi: {PUBLIC_DOI}\n"
            f"repository-code: {GITHUB_URL}\n",
            1,
        )
    else:
        # Conservative replacement if a DOI field already exists.
        lines = []
        doi_replaced = False
        repo_replaced = False
        for line in citation.splitlines():
            if line.startswith("doi:"):
                lines.append(f"doi: {PUBLIC_DOI}")
                doi_replaced = True
            elif line.startswith("repository-code:"):
                lines.append(f"repository-code: {GITHUB_URL}")
                repo_replaced = True
            else:
                lines.append(line)
        if doi_replaced and not repo_replaced:
            insert_at = next(
                (i + 1 for i, line in enumerate(lines) if line.startswith("doi:")),
                None,
            )
            if insert_at is not None:
                lines.insert(insert_at, f"repository-code: {GITHUB_URL}")
        citation = "\n".join(lines) + "\n"

    citation_path.write_text(citation, encoding="utf-8", newline="\n")

    # Optional package audit metadata.
    audit_path = root / "release_metadata" / "PACKAGE_AUDIT.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["associated_manuscript_title"] = NEW_TITLE
        audit["public_zenodo_doi"] = PUBLIC_DOI
        audit["public_zenodo_doi_status"] = "issued"
        audit["github_repository"] = GITHUB_URL
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    # Verification of the two authoritative files.
    combined = (
        readme_path.read_text(encoding="utf-8")
        + "\n"
        + citation_path.read_text(encoding="utf-8")
    )
    if OLD_TITLE in combined:
        raise RuntimeError("The obsolete manuscript title still remains.")
    if OLD_DOI_SENTENCE in combined:
        raise RuntimeError("The obsolete pending-DOI sentence still remains.")
    if NEW_TITLE not in readme_path.read_text(encoding="utf-8"):
        raise RuntimeError("The final title is missing from README.md.")
    if NEW_TITLE not in citation_path.read_text(encoding="utf-8"):
        raise RuntimeError("The final title is missing from CITATION.cff.")

    # Regenerate inventory, excluding generated index files themselves.
    excluded_inventory = {"FILE_INVENTORY.tsv", "MANIFEST_SHA256.tsv"}
    inventory_files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and p.relative_to(root).as_posix() not in excluded_inventory
        and "__pycache__" not in p.parts
    )

    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "category"],
            delimiter="\t",
        )
        writer.writeheader()
        for path in inventory_files:
            rel = path.relative_to(root).as_posix()
            writer.writerow(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "category": classify(rel),
                }
            )

    # Regenerate SHA-256 manifest, including inventory but excluding itself.
    manifest_files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and p.relative_to(root).as_posix() != "MANIFEST_SHA256.tsv"
        and "__pycache__" not in p.parts
    )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "sha256"],
            delimiter="\t",
        )
        writer.writeheader()
        for path in manifest_files:
            rel = path.relative_to(root).as_posix()
            writer.writerow(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

    print("Repository metadata correction completed successfully.")
    print(f"Final title: {NEW_TITLE}")
    print(f"Public DOI: {PUBLIC_DOI}")
    print("Updated: README.md")
    print("Updated: CITATION.cff")
    if audit_path.exists():
        print("Updated: release_metadata/PACKAGE_AUDIT.json")
    print("Regenerated: FILE_INVENTORY.tsv")
    print("Regenerated: MANIFEST_SHA256.tsv")
    print()
    print("Next commands:")
    print("  git diff --check")
    print("  git status")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
