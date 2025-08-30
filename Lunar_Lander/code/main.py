from pathlib import Path
from config import parse_args
from utils import ensure_dir, default_output_dir
from envs import make_vec_envs
from train import train_model
from evaluate import evaluate_and_record

def main():
    args = parse_args()
    out_dir = default_output_dir(args.save_dir)
    log_dir = out_dir / "logs"
    ensure_dir(log_dir)

    train_env = make_vec_envs(1)
    eval_env = make_vec_envs(1, seed=42)

    model = train_model(args, train_env, eval_env, out_dir, log_dir)

    if args.render_video:
        evaluate_and_record(out_dir)

if __name__ == "__main__":
    main()
