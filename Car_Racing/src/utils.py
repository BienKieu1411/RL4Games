import os
from pathlib import Path
import warnings

os.environ["XDG_RUNTIME_DIR"] = "/tmp"
warnings.filterwarnings("ignore")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
