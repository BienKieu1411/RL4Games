import torch
import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

from gymnasium.wrappers import RecordVideo, ResizeObservation, GrayScaleObservation, TransformReward

from utils import ensure_dir
from envs import make_carracing_env
from models import ImprovedCNN
from callbacks import SaveOnBestRewardCallback

def train_ppo_carracing(
    total_timesteps=2000000,
    n_envs=16,
    reward_threshold=90,
    save_dir="outputs",
    render_video=True,
    n_stack=4
):
    out_dir = Path(save_dir) / "carracing_ppo"
    ensure_dir(out_dir)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "best_model").mkdir(exist_ok=True)
    (out_dir / "video").mkdir(exist_ok=True)
    log_dir = out_dir / "logs"

    train_env = DummyVecEnv([make_carracing_env(seed=i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)
    train_env = VecTransposeImage(train_env)
    train_env = VecFrameStack(train_env, n_stack=n_stack)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_carracing_env(seed=42)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(200000 // n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_carracing"
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=max(200000 // n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    save_callback = SaveOnBestRewardCallback(
        check_freq=1000,
        save_path=str(out_dir / "best_model" / "best_model")
    )

    policy_kwargs = dict(
        features_extractor_class=ImprovedCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, save_callback]
    )

    model.save(str(out_dir / "final_model"))
    VecNormalize.save(train_env, str(out_dir / "vecnormalize.pkl"))

    if render_video:
        video_path = out_dir / "carracing_video"
        ensure_dir(video_path)

        def make_video_env():
            env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
            env = ResizeObservation(env, 84)
            env = GrayScaleObservation(env, keep_dim=True)
            env = TransformReward(env, lambda r: r * 0.1)
            env = RecordVideo(env, str(video_path), episode_trigger=lambda x: True)
            env.reset(seed=0)
            return env

        env = DummyVecEnv([make_video_env])
        env = VecMonitor(env)
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=n_stack)

        env = VecNormalize.load(str(out_dir / "vecnormalize.pkl"), env)
        env.training = False
        env.norm_reward = False

        model = PPO.load(str(out_dir / "best_model" / "best_model"), env=env)

        obs = env.reset()
        done = [False]
        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = dones

        env.close()
        print(f"Video saved in: {video_path}")

if __name__ == "__main__":
    train_ppo_carracing(
        total_timesteps=3000000,
        n_envs=16,
        reward_threshold=90,
        render_video=True,
        n_stack=4
    )
