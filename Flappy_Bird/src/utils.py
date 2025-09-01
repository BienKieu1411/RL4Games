import os
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def default_output_dir(save_dir):
    out_dir = Path(save_dir) / "flappy_ppo"
    ensure_dir(out_dir)
    return out_dir
