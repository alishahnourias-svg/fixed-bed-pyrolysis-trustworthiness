# Code and reproducibility package

## Associated manuscript

*From Accuracy to Trustworthiness: Grouped Validation, Applicability-Domain Mapping, and Reliability-Aware Scenario Screening for Machine-Learning-Based Fixed-Bed Biomass Pyrolysis Yield Prediction*

**Authors:** Seyed Ali Shahnouri; Mousa Nazari (corresponding author: `nazari.mousa@mzust.ac.ir`)

**Status:** Public software release `1.0.2`.

**Zenodo concept DOI (all software versions):** [10.5281/zenodo.21176726](https://doi.org/10.5281/zenodo.21176726)

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

Processed row-level data and record-level audit outputs are deposited under restricted access at [10.5281/zenodo.21175530](https://doi.org/10.5281/zenodo.21175530).

Development repository: [https://github.com/alishahnourias-svg/fixed-bed-pyrolysis-trustworthiness](https://github.com/alishahnourias-svg/fixed-bed-pyrolysis-trustworthiness).

## Repository audit

A GitHub Actions audit is included under `.github/workflows/repository-audit.yml`. It checks Python syntax, metadata parsing, and the absence of restricted files on every push and pull request.
