"""
upload_to_hf.py — Upload BioInteract hf_space/ to HuggingFace Spaces.

Usage:
    python upload_to_hf.py

Reads HF_TOKEN from ../.env (relative to BioInteract/).
Uploads hf_space/ to https://huggingface.co/spaces/AI4deeperScience/BioInteract
"""
import os
import sys
from pathlib import Path

# Load token from .env
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN   = os.environ.get('HF_TOKEN', '')
REPO_ID    = 'AI4deeperScience/BioInteract'
SPACE_DIR  = Path(__file__).parent / 'hf_space'

if not HF_TOKEN:
    sys.exit("ERROR: HF_TOKEN not found. Set it in ../.env or as an environment variable.")

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")

api = HfApi()

# ── 1. Create or verify the Space repo ───────────────────────────────────────
print(f"[1/2] Creating/verifying Space repo: {REPO_ID} …")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type='space',
        space_sdk='gradio',
        token=HF_TOKEN,
        exist_ok=True,
        private=False,
    )
    print(f"      Space ready: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"WARNING: repo creation returned: {e}")

# ── 2. Upload all files ───────────────────────────────────────────────────────
print(f"[2/2] Uploading {SPACE_DIR} → {REPO_ID} …")
api.upload_folder(
    folder_path=str(SPACE_DIR),
    repo_id=REPO_ID,
    repo_type='space',
    token=HF_TOKEN,
    commit_message='Deploy BioInteract Gradio demo',
    ignore_patterns=['__pycache__', '*.pyc', '.DS_Store'],
)

print("\n✅ Upload complete!")
print(f"   View your Space at: https://huggingface.co/spaces/{REPO_ID}")
print(f"   Build logs:         https://huggingface.co/spaces/{REPO_ID}/logs")
