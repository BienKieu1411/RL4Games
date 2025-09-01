from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from .utils import ensure_dir
from pathlib import Path

def train_model(args, train_env, eval_env, out_dir, log_dir):
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        device="cpu",
        n_steps=10000,
        batch_size=512
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_lunar"
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=args.reward_threshold,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=checkpoint_callback,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=100_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )

    model.learn(total_timesteps=args.timesteps, callback=[eval_callback])
    model.save(str(out_dir / "final_model.zip"))
    return model
