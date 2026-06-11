"""Read the newest training or experiment log file."""
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import LOGS_DIR

patterns = ['run_*.log', 'train_*.log', '*.log']
logs = []
for pattern in patterns:
    logs.extend(glob.glob(str(LOGS_DIR / pattern)))

logs = sorted(set(logs))
if not logs:
    print("No log files found")
else:
    latest = max(logs, key=os.path.getmtime)
    print(f"Log: {latest}")
    print(f"Size: {os.path.getsize(latest)} bytes")
    print(f"Modified: {os.path.getmtime(latest)}")
    with open(latest, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print("--- Last 25 lines ---")
    for line in lines[-25:]:
        print(line.rstrip())
