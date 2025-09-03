from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def linear_schedule(initial_value):
    def func(progress):
        return initial_value * (1 - progress)
    return func
