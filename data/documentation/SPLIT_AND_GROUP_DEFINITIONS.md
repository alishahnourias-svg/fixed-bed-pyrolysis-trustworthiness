# Split and group-label definitions

## Validation regimes

- `random_kfold`: five-fold shuffled K-fold with `random_state = 42`.
- `source_group_kfold`: GroupKFold using `source_group` so that records assigned to the same literature source do not cross training and test folds.
- `feedstock_group_kfold`: GroupKFold using the exact harmonized `feedstock` label.
- `family_group_kfold`: GroupKFold using `feedstock_family`; the number of folds is reduced when fewer than five families are available.

## Group labels

- `source_group`: `reference_1` when present; otherwise reconstructed from `reference_2`.
- `source_pair_group`: string pair of `reference_1` and `reference_2`, retained for conservative sensitivity diagnostics.
- `feedstock`: exact harmonized feedstock label.
- `feedstock_family`: broader taxonomy defined in `feedstock_family_mapping.csv`.

## Leakage controls

All preprocessing is inside the scikit-learn pipeline and is fitted only on the relevant training subset. For conformal analysis, the outer test fold is never used for model fitting or calibration. The outer training fold is split into proper-training and calibration subsets. Grouped calibration is used when at least three groups are available; otherwise the code records a random-split fallback that remains confined to the outer training fold.

## Files deliberately not public

Per-record fold assignments, predictions, residuals, conformal intervals, and AD scores are held in the companion restricted record. The public package retains the executable definitions and fold-level aggregate metrics.
