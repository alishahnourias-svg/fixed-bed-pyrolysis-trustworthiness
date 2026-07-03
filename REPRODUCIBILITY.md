# Reproducibility levels

## Level 1 — public auditability

The public record exposes every scientific script, all model/protocol definitions, random seeds, group-validation rules, conformal and AD methods, fold-level metrics, and aggregate outputs.

## Level 2 — processed-data reproduction

After controlled access to `clean_audit_data_with_family.xlsx`, run Phase 7 through Phase 12-C with `tools/reproduce.py --mode processed`.

## Level 3 — full-chain reconstruction

After independently obtaining the original supplementary workbooks from the publisher, run the cleaning and harmonization scripts followed by all later phases with `--mode full`.

The public/restricted split is a rights-management boundary, not a methodological omission.
