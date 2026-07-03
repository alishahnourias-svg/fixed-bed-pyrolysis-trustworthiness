#!/usr/bin/env python3
"""
Finalize the public GitHub repository for release v1.0.0.

Run from the repository root:
    python tools/finalize_release_v1_0_0.py

This script:
- inserts the reserved restricted-data DOI;
- inserts the real GitHub repository URL;
- changes the release version from 1.0.0-rc4 to 1.0.0;
- leaves the public Zenodo DOI unset until Zenodo archives the GitHub release;
- updates release metadata;
- regenerates FILE_INVENTORY.tsv and MANIFEST_SHA256.tsv;
- runs syntax and public-file safety checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path


RESTRICTED_DOI = "10.5281/zenodo.21175530"
RESTRICTED_DOI_URL = f"https://doi.org/{RESTRICTED_DOI}"
GITHUB_URL = "https://github.com/alishahnourias-svg/fixed-bed-pyrolysis-trustworthiness"
FINAL_VERSION = "1.0.0"
RELEASE_DATE = "2026-07-03"


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(
            f"Could not find the expected text while updating {label}:\n{old}"
        )
    return text.replace(old, new)


def classify(rel: str) -> str:
    if rel.startswith(".github/"):
        return "github_automation"
    if rel.startswith("code/") or rel.startswith("tools/"):
        return "code_or_tool"
    if rel.startswith("data/") or rel.startswith("release_metadata/"):
        return "documentation" if rel.startswith("data/") else "release_metadata"
    if rel.startswith("figures/"):
        return "figure"
    if rel.startswith("results/"):
        return "aggregate_result"
    return "root_metadata"


def main() -> int:
    root = Path.cwd().resolve()

    required = [
        root / "README.md",
        root / "CITATION.cff",
        root / "VERSION",
        root / "FILE_INVENTORY.tsv",
        root / "MANIFEST_SHA256.tsv",
        root / "release_metadata" / "PACKAGE_AUDIT.json",
        root / "release_metadata" / "DATA_AND_CODE_AVAILABILITY_TEMPLATE.md",
        root / "release_metadata" / "FINAL_PUBLIC_RELEASE_CHECKLIST.md",
        root / "tools" / "audit_public_repository.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run this script from the repository root. Missing:\n- "
            + "\n- ".join(missing)
        )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")

    readme = replace_required(
        readme,
        "**Status:** GitHub-ready release candidate `1.0.0-rc4`. Replace DOI placeholders and insert the final GitHub URL and the two Zenodo DOIs, then change the version to `1.0.0` immediately before the DOI-bearing release.",
        "**Status:** Public software release `1.0.0`, prepared for DOI-bearing archival through the Zenodo–GitHub integration.",
        "README status",
    )

    readme = replace_required(
        readme,
        "Processed row-level data and record-level audit outputs: `[RESTRICTED ZENODO DOI]`.\n\nPublic code and aggregate-output DOI: `[PUBLIC ZENODO DOI]`.",
        f"Processed row-level data and record-level audit outputs are deposited under restricted access at [{RESTRICTED_DOI}]({RESTRICTED_DOI_URL}).\n\nDevelopment repository: [{GITHUB_URL}]({GITHUB_URL}).\n\nThe version-specific public software DOI will be minted by Zenodo after GitHub release `v1.0.0` is published.",
        "README companion records",
    )

    readme_path.write_text(readme, encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------
    # CITATION.cff
    # Do not add a public DOI before Zenodo creates it from the release.
    # ------------------------------------------------------------------
    cff_path = root / "CITATION.cff"
    cff = cff_path.read_text(encoding="utf-8")
    cff = replace_required(cff, "version: 1.0.0-rc4", f"version: {FINAL_VERSION}", "CITATION version")
    cff = replace_required(cff, "date-released: '2026-07-03'", f"date-released: '{RELEASE_DATE}'", "CITATION date")

    if "repository-code:" not in cff:
        cff = replace_required(
            cff,
            "license: MIT\n",
            f"license: MIT\nrepository-code: {GITHUB_URL}\n",
            "CITATION repository URL",
        )

    cff_path.write_text(cff, encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------
    # VERSION
    # ------------------------------------------------------------------
    (root / "VERSION").write_text(FINAL_VERSION + "\n", encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------
    # Package audit
    # ------------------------------------------------------------------
    audit_path = root / "release_metadata" / "PACKAGE_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["version"] = FINAL_VERSION
    audit["github_repository"] = GITHUB_URL
    audit["restricted_record_doi"] = RESTRICTED_DOI
    audit["public_zenodo_doi_status"] = "pending GitHub release archival"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # ------------------------------------------------------------------
    # Availability statement template
    # ------------------------------------------------------------------
    availability_path = (
        root / "release_metadata" / "DATA_AND_CODE_AVAILABILITY_TEMPLATE.md"
    )
    availability = availability_path.read_text(encoding="utf-8")
    availability = availability.replace(
        "[RESTRICTED DOI]",
        RESTRICTED_DOI,
    )
    availability = availability.replace(
        "[GITHUB URL]",
        GITHUB_URL,
    )
    availability = availability.replace(
        "[PUBLIC DOI]",
        "[PUBLIC DOI TO BE INSERTED AFTER ZENODO ARCHIVES GITHUB RELEASE v1.0.0]",
    )
    availability_path.write_text(
        availability, encoding="utf-8", newline="\n"
    )

    # ------------------------------------------------------------------
    # Checklist
    # ------------------------------------------------------------------
    checklist_path = (
        root / "release_metadata" / "FINAL_PUBLIC_RELEASE_CHECKLIST.md"
    )
    checklist = checklist_path.read_text(encoding="utf-8")
    checklist = checklist.replace(
        "- [ ] GitHub repository URL inserted.",
        "- [x] GitHub repository URL inserted.",
    )
    checklist = checklist.replace(
        "- [ ] Restricted-record DOI reserved and inserted.",
        "- [x] Restricted-record DOI reserved and inserted.",
    )
    checklist = checklist.replace(
        "- [ ] Version changed from GitHub-ready release candidate to `1.0.0`.",
        "- [x] Version changed from GitHub-ready release candidate to `1.0.0`.",
    )
    checklist_path.write_text(checklist, encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------
    # Syntax checks
    # ------------------------------------------------------------------
    python_files = sorted((root / "code").rglob("*.py")) + sorted(
        (root / "tools").rglob("*.py")
    )
    for path in python_files:
        py_compile.compile(str(path), doraise=True)

    # Public-file safety audit
    subprocess.run(
        [sys.executable, str(root / "tools" / "audit_public_repository.py")],
        cwd=root,
        check=True,
    )

    # Remove generated __pycache__ directories before inventory/checksums.
    for cache_dir in sorted(root.rglob("__pycache__"), reverse=True):
        for item in cache_dir.iterdir():
            item.unlink()
        cache_dir.rmdir()

    # ------------------------------------------------------------------
    # FILE_INVENTORY.tsv
    # Exclude the two generated index files to avoid recursion.
    # ------------------------------------------------------------------
    excluded_inventory = {"FILE_INVENTORY.tsv", "MANIFEST_SHA256.tsv"}
    files_for_inventory = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.relative_to(root).as_posix() not in excluded_inventory
    )

    inventory_path = root / "FILE_INVENTORY.tsv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "category"],
            delimiter="\t",
        )
        writer.writeheader()
        for path in files_for_inventory:
            rel = path.relative_to(root).as_posix()
            writer.writerow(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "category": classify(rel),
                }
            )

    # ------------------------------------------------------------------
    # MANIFEST_SHA256.tsv
    # Include FILE_INVENTORY.tsv, but exclude the manifest itself.
    # ------------------------------------------------------------------
    files_for_manifest = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.relative_to(root).as_posix() != "MANIFEST_SHA256.tsv"
    )

    manifest_path = root / "MANIFEST_SHA256.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "sha256"],
            delimiter="\t",
        )
        writer.writeheader()
        for path in files_for_manifest:
            rel = path.relative_to(root).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            writer.writerow(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )

    print()
    print("Finalization completed successfully.")
    print(f"Version: {FINAL_VERSION}")
    print(f"GitHub: {GITHUB_URL}")
    print(f"Restricted DOI: {RESTRICTED_DOI}")
    print("Public Zenodo DOI: pending release archival")
    print(f"Python files compiled: {len(python_files)}")
    print("FILE_INVENTORY.tsv and MANIFEST_SHA256.tsv regenerated.")
    print()
    print("Next review command:")
    print("  git status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
