from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

def train_model(train_env, eval_env, log_dir, out_dir, timesteps=1_000_000, reward_threshold=100.0, device="cuda"):
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=device,
        n_steps=10000,
        batch_size=512
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_flappy"
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=checkpoint_callback,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )

    model.learn(total_timesteps=timesteps, callback=[eval_callback])
    model.save(str(out_dir / "final_model.zip"))
    return model