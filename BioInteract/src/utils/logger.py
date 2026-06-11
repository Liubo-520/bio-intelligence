"""
logger.py — Logging and experiment tracking utilities.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

from .paths import LOGS_DIR, RESULTS_DIR, resolve_project_path


def setup_logger(name: str, log_dir: str = 'logs',
                 level: int = logging.INFO) -> logging.Logger:
    """Set up a logger that writes to both console and file."""
    log_dir_path = resolve_project_path(log_dir)
    os.makedirs(log_dir_path, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir_path, f'{name}_{timestamp}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                   datefmt='%H:%M:%S')
    ch.setFormatter(ch_format)
    logger.addHandler(ch)
    
    # file handler — force flush after every log record for real-time visibility
    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    fh = FlushFileHandler(log_file)
    fh.setLevel(level)
    fh_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fh_format)
    logger.addHandler(fh)
    
    return logger


class ExperimentTracker:
    """Simple experiment result tracker that saves to JSON."""
    
    def __init__(self, save_dir: str = 'results'):
        self.save_dir = resolve_project_path(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.records = []
    
    def log_experiment(self, config: dict, metrics: dict,
                       split_type: str, dataset: str, notes: str = ''):
        record = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset,
            'split_type': split_type,
            'config': config,
            'metrics': metrics,
            'notes': notes,
        }
        self.records.append(record)
        
        # save immediately
        save_path = os.path.join(self.save_dir, 'experiments.json')
        with open(save_path, 'w') as f:
            json.dump(self.records, f, indent=2)
    
    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.save_dir, 'experiments.json')
        else:
            path = resolve_project_path(path)
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.records = json.load(f)
