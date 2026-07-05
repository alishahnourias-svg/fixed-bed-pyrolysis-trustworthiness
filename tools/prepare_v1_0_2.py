#!/usr/bin/env python3
"""
Prepare the fixed-bed-pyrolysis-trustworthiness repository for release v1.0.2.

Run from the repository root:
    python tools/prepare_v1_0_2.py

This script:
- rewrites README.md with the final manuscript title and stable Zenodo concept DOI;
- rewrites CITATION.cff with the final title, version 1.0.2, and concept DOI;
- updates VERSION and selected release metadata;
- replaces the obsolete v1.0.0 DOI in current text metadata;
- removes one-off maintenance scripts that contain obsolete metadata;
- regenerates FILE_INVENTORY.tsv and MANIFEST_SHA256.tsv;
- verifies that the obsolete manuscript title and obsolete DOI are absent
  from the current repository files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

FINAL_TITLE = (
    "From Accuracy to Trustworthiness: Grouped Validation, "
    "Applicability-Domain Mapping, and Reliability-Aware Scenario Screening "
    "for Machine-Learning-Based Fixed-Bed Biomass Pyrolysis Yield Prediction"
)

PACKAGE_TITLE = f"Code and Reproducibility Package for: {FINAL_TITLE}"

OBSOLETE_TITLE = (
    "From Accuracy to Trustworthiness: Grouped Validation, "
    "Applicability-Domain Mapping, and Reliability-Aware Optimization "
    "for Machine-Learning-Based Fixed-Bed Biomass Pyrolysis Yield Prediction"
)

CONCEPT_DOI = "10.5281/zenodo.21176726"
OBSOLETE_VERSION_DOI = "10.5281/zenodo.21176726"
RESTRICTED_DOI = "10.5281/zenodo.21175530"
GITHUB_URL = (
    "https://github.com/alishahnourias-svg/"
    "fixed-bed-pyrolysis-trustworthiness"
)
VERSION = "1.0.2"
RELEASE_DATE = "2026-07-05"

TEXT_SUFFIXES = {
    ".md", ".cff", ".json", ".txt", ".yml", ".yaml", ".toml",
    ".tsv", ".csv", ".py"
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run this script from the repository root."
        )


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


def replace_current_text_metadata(root: Path) -> list[str]:
    changed: list[str] = []

    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        updated = text.replace(OBSOLETE_TITLE, FINAL_TITLE)
        updated = updated.replace(OBSOLETE_VERSION_DOI, CONCEPT_DOI)

        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="\n")
            changed.append(path.relative_to(root).as_posix())

    return changed


def main() -> int:
    root = Path.cwd().resolve()

    readme_path = root / "README.md"
    citation_path = root / "CITATION.cff"
    version_path = root / "VERSION"
    inventory_path = root / "FILE_INVENTORY.tsv"
    manifest_path = root / "MANIFEST_SHA256.tsv"

    for path in (
        readme_path,
        citation_path,
        version_path,
        inventory_path,
        manifest_path,
    ):
        require(path)

    # Remove one-off metadata-maintenance scripts from the final scientific release.
    for rel in (
        "tools/finalize_release_v1_0_0.py",
        "tools/fix_live_repository_metadata.py",
    ):
        path = root / rel
        if path.exists():
            path.unlink()

    # Remove generated caches.
    for cache_dir in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir)

    # Replace obsolete title and obsolete v1.0.0 DOI in current textual metadata.
    changed_files = replace_current_text_metadata(root)

    # Rewrite README.md deterministically.
    readme = f"""# Code and reproducibility package

## Associated manuscript

*{FINAL_TITLE}*

**Authors:** Seyed Ali Shahnouri; Mousa Nazari (corresponding author: `nazari.mousa@mzust.ac.ir`)

**Status:** Public software release `{VERSION}`.

**Zenodo concept DOI (all software versions):** [{CONCEPT_DOI}](https://doi.org/{CONCEPT_DOI})

The concept DOI above always resolves to the latest archived software release. For exact reproducibility, cite the version-specific DOI corresponding to the release used.

## Scope of this public record

This public package contains the complete Python workflow, environment files, data dictionary, feedstock-family mapping, feature-protocol and split definitions, fold-level metrics, aggregate uncertainty and applicability-domain diagnostics, eligibility summaries, final reliability-aware screening windows, manuscript summary tables, aggregate figure data, and publication figures.

It deliberately excludes all row-level processed datasets, per-record predictions, per-record conformal intervals, per-record applicability-domain scores, and candidate-level scenario tables. Those files are assigned to a separate restricted Zenodo record. The original supplementary workbooks from Abaei et al. are not redistributed in either record.

## Reproduction

Create the environment with either:

```bash
conda env create -f environment.yml
conda activate fixed-bed-pyrolysis-trustworthiness
```

or:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

To reproduce Phase 7 onward after authorised access to the restricted processed dataset:

```bash
python tools/reproduce.py --mode processed --data-dir PATH_TO_RESTRICTED_DATA --workspace reproduction_run
```

To reproduce the complete chain from locally obtained source supplementary files:

```bash
python tools/reproduce.py --mode full --raw-dir PATH_TO_SOURCE_FILES --workspace reproduction_full
```

The source directory must contain `SI_Data.xlsx` and `SI_Data_References.xlsx`.

## Licenses

Code is MIT licensed. Author-generated tables, aggregate outputs, figures, figure data, and original documentation are CC BY 4.0. See `LICENSES.md`.

## Companion restricted record

Processed row-level data and record-level audit outputs are deposited under restricted access at [{RESTRICTED_DOI}](https://doi.org/{RESTRICTED_DOI}).

Development repository: [{GITHUB_URL}]({GITHUB_URL}).

## Repository audit

A GitHub Actions audit is included under `.github/workflows/repository-audit.yml`. It checks Python syntax, metadata parsing, and the absence of restricted files on every push and pull request.
"""
    readme_path.write_text(readme, encoding="utf-8", newline="\n")

    # Rewrite CITATION.cff deterministically.
    citation = f"""cff-version: 1.2.0
message: "If you use this software or its aggregate analytical outputs, cite the archived release and the associated manuscript."
title: "{PACKAGE_TITLE}"
type: software
version: {VERSION}
date-released: "{RELEASE_DATE}"
doi: "{CONCEPT_DOI}"
repository-code: "{GITHUB_URL}"
license: MIT
authors:
  - family-names: Shahnouri
    given-names: Seyed Ali
    affiliation: "Department of Biosystems Engineering, Sari University of Agricultural Sciences and Natural Resources, Sari, Iran"
  - family-names: Nazari
    given-names: Mousa
    affiliation: "Department of Computer Science, Faculty of Science, University of Science and Technology of Mazandaran, Behshahr, Iran"
    email: "nazari.mousa@mzust.ac.ir"
contact:
  - family-names: Nazari
    given-names: Mousa
    email: "nazari.mousa@mzust.ac.ir"
    affiliation: "Department of Computer Science, Faculty of Science, University of Science and Technology of Mazandaran, Behshahr, Iran"
keywords:
  - biomass pyrolysis
  - machine learning
  - grouped validation
  - conformal prediction
  - applicability domain
  - reliability-aware scenario screening
abstract: "Python workflow and aggregate reproducibility outputs for grouped validation, conformal calibration, applicability-domain mapping, and reliability-aware scenario screening of fixed-bed biomass pyrolysis yield models."
"""
    citation_path.write_text(citation, encoding="utf-8", newline="\n")

    # VERSION
    version_path.write_text(VERSION + "\n", encoding="utf-8", newline="\n")

    # Optional package audit.
    audit_path = root / "release_metadata" / "PACKAGE_AUDIT.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["version"] = VERSION
        audit["associated_manuscript_title"] = FINAL_TITLE
        audit["zenodo_concept_doi"] = CONCEPT_DOI
        audit["restricted_record_doi"] = RESTRICTED_DOI
        audit["github_repository"] = GITHUB_URL
        audit["release_status"] = "prepared for v1.0.2 archival"
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    # Verify current repository text.
    obsolete_title_hits: list[str] = []
    obsolete_doi_hits: list[str] = []

    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        rel = path.relative_to(root).as_posix()
        if OBSOLETE_TITLE in text:
            obsolete_title_hits.append(rel)
        if OBSOLETE_VERSION_DOI in text:
            obsolete_doi_hits.append(rel)

    if obsolete_title_hits:
        raise RuntimeError(
            "Obsolete manuscript title remains in:\n- "
            + "\n- ".join(obsolete_title_hits)
        )

    if obsolete_doi_hits:
        raise RuntimeError(
            "Obsolete v1.0.0 DOI remains in:\n- "
            + "\n- ".join(obsolete_doi_hits)
        )

    # Regenerate file inventory.
    excluded_inventory = {"FILE_INVENTORY.tsv", "MANIFEST_SHA256.tsv"}
    inventory_files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p.relative_to(root).as_posix() not in excluded_inventory
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

    # Regenerate SHA-256 manifest.
    manifest_files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p.relative_to(root).as_posix() != "MANIFEST_SHA256.tsv"
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

    print("v1.0.2 repository preparation completed successfully.")
    print(f"Version: {VERSION}")
    print(f"Final title: {FINAL_TITLE}")
    print(f"Zenodo concept DOI: {CONCEPT_DOI}")
    print(f"Restricted data DOI: {RESTRICTED_DOI}")
    print(f"Text metadata files normalized: {len(changed_files)}")
    print("Obsolete title remaining: 0")
    print("Obsolete v1.0.0 DOI remaining: 0")
    print("FILE_INVENTORY.tsv and MANIFEST_SHA256.tsv regenerated.")
    print()
    print("Next commands:")
    print("  git diff --check")
    print("  git status")


if __name__ == "__main__":
    raise SystemExit(main())
