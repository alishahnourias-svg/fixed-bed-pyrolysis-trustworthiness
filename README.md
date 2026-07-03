# Code and reproducibility package

## Associated manuscript

*From Accuracy to Trustworthiness: Grouped Validation, Applicability-Domain Mapping, and Reliability-Aware Optimization for Machine-Learning-Based Fixed-Bed Biomass Pyrolysis Yield Prediction*

**Authors:** Seyed Ali Shahnouri; Mousa Nazari (corresponding author: `nazari.mousa@mzust.ac.ir`)

**Status:** Public software release `1.0.0`, prepared for DOI-bearing archival through the Zenodo–GitHub integration.

## Scope of this public record

This public package contains the complete Python workflow, environment files, data dictionary, feedstock-family mapping, feature-protocol and split definitions, fold-level metrics, aggregate uncertainty and applicability-domain diagnostics, eligibility summaries, final reliability-aware screening windows, manuscript summary tables, aggregate figure data, and publication figures.

It deliberately excludes all row-level processed datasets, per-record predictions, per-record conformal intervals, per-record AD scores, and candidate-level scenario tables. Those files are assigned to a separate restricted Zenodo record. The original supplementary workbooks from Abaei et al. are not redistributed in either record.

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
python tools/reproduce.py --mode processed --data-dir PATH_TO_RESTRICTED_DATA \
  --workspace reproduction_run
```

To reproduce the complete chain from locally obtained source supplementary files:

```bash
python tools/reproduce.py --mode full --raw-dir PATH_TO_SOURCE_FILES \
  --workspace reproduction_full
```

The source directory must contain `SI_Data.xlsx` and `SI_Data_References.xlsx`.

## Licenses

Code is MIT licensed. Author-generated tables, aggregate outputs, figures, figure data, and original documentation are CC BY 4.0. See `LICENSES.md`. `CITATION.cff` records MIT as the primary software license; the Zenodo record must additionally list CC BY 4.0 for the specified non-code files.

## Companion restricted record

Processed row-level data and record-level audit outputs are deposited under restricted access at [10.5281/zenodo.21175530](https://doi.org/10.5281/zenodo.21175530).

Development repository: [https://github.com/alishahnourias-svg/fixed-bed-pyrolysis-trustworthiness](https://github.com/alishahnourias-svg/fixed-bed-pyrolysis-trustworthiness).

The version-specific public software DOI will be minted by Zenodo after GitHub release `v1.0.0` is published.


## Publish this package to GitHub

The recommended repository name is:

```text
fixed-bed-pyrolysis-trustworthiness
```

On Windows PowerShell, from the repository root, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\tools\publish_to_github.ps1
```

The script initializes Git, checks for forbidden row-level files, creates the first commit, and either uses GitHub CLI to create/push the public repository or prints the exact fallback commands. See `release_metadata/GITHUB_PUBLICATION_GUIDE.md`.

A GitHub Actions audit is included under `.github/workflows/repository-audit.yml`. It checks Python syntax, metadata parsing, and the absence of restricted files on every push and pull request.
