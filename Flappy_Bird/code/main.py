import argparse
import warnings
from utils import default_output_dir, ensure_dir
from envs import make_vec_envs
from train import train_model
from render_video import render_model

import os
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
warnings.filterwarnings("ignore")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--reward_threshold", type=float, default=100.0)
    p.add_argument("--render_video", action="store_true")
    return p.parse_args(args=["--render_video"])

def main():
    args = parse_args()
    out_dir = default_output_dir(args.save_dir)
    log_dir = out_dir / "logs"
    ensure_dir(log_dir)

    train_env = make_vec_envs(1)
    eval_env = make_vec_envs(1, seed=42)

    model = train_model(train_env, eval_env, log_dir, out_dir, args.timesteps, args.reward_threshold)

    if args.render_video:
        best_model_path = out_dir / "best_model" / "best_model.zip"
        video_path = out_dir / "flappy_video"
        render_model(best_model_path, video_path)

if __name__ == "__main__":
    main()