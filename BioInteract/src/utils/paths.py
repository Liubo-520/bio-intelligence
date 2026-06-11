from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

CONFIGS_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'
RUNS_DIR = PROJECT_ROOT / 'runs'

SUBMISSION_DIR = WORKSPACE_ROOT / 'submission'
MANUSCRIPT_DIR = SUBMISSION_DIR / 'manuscript'
MANUSCRIPT_FIGURES_DIR = MANUSCRIPT_DIR / 'figures'
DOCS_DIR = WORKSPACE_ROOT / 'docs'


def resolve_project_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_workspace_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path