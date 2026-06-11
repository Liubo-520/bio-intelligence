"""
dock_validate.py — Prepare docking jobs for top predicted pairs and run Vina when possible.

Usage:
    python -m src.tools.dock_validate --predictions results/top_predictions.csv --pdb_dir path/to/pdbs
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils.paths import DATA_DIR, RESULTS_DIR, resolve_project_path


def load_smiles_lookup(dataset: str) -> dict[str, str]:
    csv_path = DATA_DIR / 'raw' / dataset / 'drug_smiles.csv'
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path)
    return dict(zip(frame['drug_id'].astype(str), frame['smiles'].astype(str)))


def select_top_predictions(predictions_path: Path, top_k: int) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path)
    if 'prediction' in frame.columns:
        frame = frame.sort_values('prediction', ascending=False)
    return frame.head(top_k).reset_index(drop=True)


def parse_vina_log(log_file: Path) -> dict:
    result = {'best_affinity': None, 'modes': []}
    if not log_file.exists():
        return result

    parsing = False
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith('mode |   affinity'):
                parsing = True
                continue
            if parsing and stripped and stripped[0].isdigit():
                parts = stripped.split()
                if len(parts) >= 4:
                    mode = {
                        'mode': int(parts[0]),
                        'affinity_kcal': float(parts[1]),
                        'rmsd_lb': float(parts[2]),
                        'rmsd_ub': float(parts[3]),
                    }
                    result['modes'].append(mode)

    if result['modes']:
        result['best_affinity'] = result['modes'][0]['affinity_kcal']
    return result


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def maybe_build_ligand_pdbqt(smiles_file: Path, output_dir: Path, obabel_cmd: str) -> Path | None:
    if shutil.which(obabel_cmd) is None:
        return None
    ligand_pdbqt = output_dir / 'ligand.pdbqt'
    result = run_command([obabel_cmd, str(smiles_file), '-O', str(ligand_pdbqt), '--gen3d'])
    if result.returncode != 0 or not ligand_pdbqt.exists():
        return None
    return ligand_pdbqt


def maybe_prepare_receptor(receptor_file: Path) -> Path | None:
    if receptor_file.suffix.lower() == '.pdbqt' and receptor_file.exists():
        return receptor_file
    return None


def maybe_run_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_dir: Path,
    vina_cmd: str,
    center: tuple[float, float, float] | None,
    size: tuple[float, float, float],
) -> dict:
    output_pdbqt = output_dir / 'result.pdbqt'
    log_file = output_dir / 'vina.log'

    if shutil.which(vina_cmd) is None:
        return {'status': 'skipped_missing_vina'}
    if center is None:
        return {'status': 'prepared_only_missing_center'}

    command = [
        vina_cmd,
        '--receptor', str(receptor_pdbqt),
        '--ligand', str(ligand_pdbqt),
        '--out', str(output_pdbqt),
        '--log', str(log_file),
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x', str(size[0]),
        '--size_y', str(size[1]),
        '--size_z', str(size[2]),
    ]
    result = run_command(command, cwd=output_dir)
    summary = parse_vina_log(log_file)
    summary['status'] = 'completed' if result.returncode == 0 else 'failed'
    summary['stdout'] = result.stdout[-4000:]
    summary['stderr'] = result.stderr[-4000:]
    return summary


def main():
    parser = argparse.ArgumentParser(description='Validate top predictions with docking inputs or Vina')
    parser.add_argument('--predictions', type=str, default=str(RESULTS_DIR / 'top_predictions.csv'))
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=str(RESULTS_DIR / 'docking'))
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--vina_cmd', type=str, default='vina')
    parser.add_argument('--obabel_cmd', type=str, default='obabel')
    parser.add_argument('--center_x', type=float, default=None)
    parser.add_argument('--center_y', type=float, default=None)
    parser.add_argument('--center_z', type=float, default=None)
    parser.add_argument('--size_x', type=float, default=20.0)
    parser.add_argument('--size_y', type=float, default=20.0)
    parser.add_argument('--size_z', type=float, default=20.0)
    args = parser.parse_args()

    predictions_path = resolve_project_path(args.predictions)
    pdb_dir = resolve_project_path(args.pdb_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_lookup = load_smiles_lookup(args.dataset)
    top_predictions = select_top_predictions(predictions_path, args.top_k)
    center = None
    if None not in (args.center_x, args.center_y, args.center_z):
        center = (args.center_x, args.center_y, args.center_z)
    size = (args.size_x, args.size_y, args.size_z)

    all_results = []
    for idx, row in top_predictions.iterrows():
        drug_id = str(row.get('drug_id', f'D{idx:04d}'))
        target_id = str(row.get('target_id', f'T{idx:04d}'))
        smiles = str(row.get('smiles', smiles_lookup.get(drug_id, '')))

        pair_dir = output_dir / f'{target_id}_{idx:03d}'
        pair_dir.mkdir(parents=True, exist_ok=True)
        smiles_file = pair_dir / 'ligand.smi'
        smiles_file.write_text(smiles + '\n', encoding='utf-8')

        receptor_file = None
        for suffix in ('.pdbqt', '.pdb'):
            candidate = pdb_dir / f'{target_id}{suffix}'
            if candidate.exists():
                receptor_file = candidate
                break

        record = {
            'rank': idx + 1,
            'drug_id': drug_id,
            'target_id': target_id,
            'prediction': float(row['prediction']) if 'prediction' in row else None,
            'pair_dir': str(pair_dir),
            'receptor_file': str(receptor_file) if receptor_file else None,
            'ligand_smiles_file': str(smiles_file),
        }

        metadata_path = pair_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)

        if not smiles:
            record['status'] = 'skipped_missing_smiles'
            all_results.append(record)
            continue
        if receptor_file is None:
            record['status'] = 'skipped_missing_receptor'
            all_results.append(record)
            continue

        ligand_pdbqt = maybe_build_ligand_pdbqt(smiles_file, pair_dir, args.obabel_cmd)
        receptor_pdbqt = maybe_prepare_receptor(receptor_file)

        if ligand_pdbqt is None:
            record['status'] = 'prepared_only_missing_ligand_pdbqt'
            all_results.append(record)
            continue
        if receptor_pdbqt is None:
            record['status'] = 'prepared_only_missing_receptor_pdbqt'
            all_results.append(record)
            continue

        docking_result = maybe_run_vina(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            output_dir=pair_dir,
            vina_cmd=args.vina_cmd,
            center=center,
            size=size,
        )
        record.update(docking_result)
        all_results.append(record)

    summary_path = output_dir / 'docking_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    print(f'Docking results saved to {summary_path}')


if __name__ == '__main__':
    main()