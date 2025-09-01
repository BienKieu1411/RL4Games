import os
from pathlib import Path
import warnings

os.environ["XDG_RUNTIME_DIR"] = "/tmp"
warnings.filterwarnings("ignore")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def default_output_dir(save_dir):
    out_dir = Path(save_dir) / "lunarlander_ppo"
    ensure_dir(out_dir)
    return out_dir
