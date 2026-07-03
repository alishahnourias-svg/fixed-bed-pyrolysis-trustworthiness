#!/usr/bin/env python3
"""Run the official workflow in an isolated flat workspace.

The public repository intentionally contains no row-level data.

processed mode:
    Requires --data-dir pointing to an authorised copy of
    clean_audit_data_with_family.xlsx (normally obtained from the companion
    restricted record). Runs Phase 7 through Phase 12-C.

full mode:
    Requires --raw-dir pointing to locally obtained SI_Data.xlsx and
    SI_Data_References.xlsx. Runs the entire chain from 00a onward.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    '00a_data_cleaning_and_harmonization.py',
    '00b_feedstock_family_and_feature_protocols.py',
    '01_phase7_grouped_validation.py',
    '02_phase8_official_model_selection.py',
    '03_phase9_conformal_uncertainty_calibration.py',
    '04_phase10_applicability_domain_mapping.py',
    '05_phase11b1_eligibility_and_scenario_space.py',
    '06_phase11b2_reliability_aware_candidate_ranking.py',
    '07_phase11b3_v2_refine_scenario_outputs_official.py',
    '08_phase12a_build_results_master_tables.py',
    '09_phase12b_prepare_manuscript_tables_and_figures.py',
    '10_phase12c_refine_figures_and_results_skeleton.py',
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('processed','full'), required=True)
    p.add_argument('--workspace', type=Path, default=Path('reproduction_run'))
    p.add_argument('--data-dir', type=Path, help='Directory containing clean_audit_data_with_family.xlsx')
    p.add_argument('--raw-dir', type=Path, help='Directory containing the two source supplementary workbooks')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()

def main():
    a = parse_args()
    repo = Path(__file__).resolve().parents[1]
    ws = a.workspace.resolve()
    if ws.exists():
        if not a.overwrite:
            raise FileExistsError(f'{ws} exists; use --overwrite or another path.')
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy2(repo/'code'/name, ws/name)

    if a.mode == 'processed':
        if a.data_dir is None:
            raise ValueError('--data-dir is required for processed mode.')
        src = a.data_dir.resolve()/'clean_audit_data_with_family.xlsx'
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, ws/src.name)
        run_list = SCRIPTS[2:]
    else:
        if a.raw_dir is None:
            raise ValueError('--raw-dir is required for full mode.')
        for name in ('SI_Data.xlsx','SI_Data_References.xlsx'):
            src = a.raw_dir.resolve()/name
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, ws/name)
        run_list = SCRIPTS

    log_path = ws/'reproduction_run.log'
    with log_path.open('w', encoding='utf-8') as log:
        for name in run_list:
            print(f'> {sys.executable} {name}')
            log.write(f'\n===== {name} =====\n')
            log.flush()
            if a.dry_run:
                continue
            r = subprocess.run([sys.executable, name], cwd=ws, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                raise RuntimeError(f'{name} failed. See {log_path}')
    print(f'Workspace: {ws}')
    print(f'Log: {log_path}')

if __name__ == '__main__':
    main()
