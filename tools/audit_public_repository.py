#!/usr/bin/env python3
"""Fail if restricted or source-level files are present in the public repository."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_EXACT = ['SI_Data.xlsx', 'SI_Data_References.xlsx', 'requirements_full.txt', 'code.pdf', 'phase7_v2_prediction_results.xlsx', 'phase9_prediction_intervals.xlsx', 'phase10_ad_scores.xlsx', 'phase10_predictions_with_ad.xlsx', 'phase10_top_out_of_domain_errors.xlsx', 'phase11_eligibility_audit.xlsx', 'phase11_eligible_entities.xlsx', 'phase11_ineligible_entities.xlsx', 'phase11_scenario_space_definitions.xlsx', 'phase11b2_candidate_scenarios.xlsx', 'phase11b2_top_ranked_scenarios.xlsx', 'phase11b3_v2_top_scenario_examples.xlsx']
FORBIDDEN_PATTERNS = ['clean_audit_data*.xlsx', 'biochar_dataset*.xlsx', 'bio_oil_dataset*.xlsx', 'paired_dataset*.xlsx']

problems = []
for name in FORBIDDEN_EXACT:
    for path in ROOT.rglob(name):
        if '.git' not in path.parts:
            problems.append(path.relative_to(ROOT).as_posix())
for pattern in FORBIDDEN_PATTERNS:
    for path in ROOT.rglob(pattern):
        if '.git' not in path.parts:
            problems.append(path.relative_to(ROOT).as_posix())

if problems:
    print('Forbidden files detected in public repository:')
    for item in sorted(set(problems)):
        print(f' - {item}')
    sys.exit(1)

print('Public-repository file audit passed: no restricted/source files detected.')
