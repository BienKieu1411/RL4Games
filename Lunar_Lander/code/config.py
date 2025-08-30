import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--reward_threshold", type=float, default=250.0)
    p.add_argument("--render_video", action="store_true")
    return p.parse_args(args=["--render_video"])  
