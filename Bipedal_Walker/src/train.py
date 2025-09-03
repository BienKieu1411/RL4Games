import os
import warnings
from pathlib import Path
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium.wrappers import RecordVideo

from utils import ensure_dir, linear_schedule
from envs import make_vec_envs
from callbacks import SaveVecNormalizeCallback

os.environ["XDG_RUNTIME_DIR"] = "/tmp"
warnings.filterwarnings("ignore")

def train_ppo_bipedal(
    total_timesteps=20_000_000,
    n_envs=16,
    save_dir="outputs_improved",
    render_video=True
):
    out_dir = Path(save_dir) / "bipedalwalker_ppo_advanced"
    ensure_dir(out_dir)
    ensure_dir(out_dir / "checkpoints")
    ensure_dir(out_dir / "best_model")
    ensure_dir(out_dir / "video")
    ensure_dir(out_dir / "logs")

    train_env = make_vec_envs(n_envs=n_envs, seed=0, difficulty=0)
    eval_env = make_vec_envs(n_envs=1, seed=42, difficulty=1)
    eval_env.norm_reward = False

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=nn.ReLU,
        log_std_init=-0.5,
        ortho_init=True,
        share_features_extractor=False
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=2,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,
        learning_rate=linear_schedule(3e-4),
        tensorboard_log=str(out_dir / "logs"),
        device="auto",
        policy_kwargs=policy_kwargs,
        seed=42
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=max(1000000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000000 // n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_bipedal"
    )
    
    vec_normalize_callback = SaveVecNormalizeCallback(save_path=str(out_dir / "vecnormalize.pkl"))

    callbacks = [eval_callback, checkpoint_callback, vec_normalize_callback]

    # Training easy env
    model.learn(
        total_timesteps=total_timesteps // 3,
        callback=callbacks,
        tb_log_name="ppo_bipedal_easy"
    )

    train_env.close()
    train_env = make_vec_envs(n_envs=n_envs, seed=0, difficulty=1)
    model.set_env(train_env)
    
    # Training hardcore env
    model.learn(
        total_timesteps=total_timesteps // 3 * 2,
        callback=callbacks,
        tb_log_name="ppo_bipedal_hardcore"
    )

    model.save(str(out_dir / "final_model"))
    train_env.save(str(out_dir / "vecnormalize.pkl"))

    train_env.close()
    eval_env.close()

    if render_video:
        env = gym.make("BipedalWalkerHardcore-v3", render_mode="rgb_array")
        env = RecordVideo(env, str(out_dir / "video"), 
                         episode_trigger=lambda x: True,
                         name_prefix="bipedalwalker")
        
        best_model_path = out_dir / "best_model" / "best_model.zip"
        model = PPO.load(str(best_model_path), device="auto")
        
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        vec_normalize = VecNormalize.load(str(out_dir / "vecnormalize.pkl"), 
                                         DummyVecEnv([lambda: gym.make("BipedalWalkerHardcore-v3")]))
        vec_normalize.training = False
        vec_normalize.norm_reward = False

        obs, _ = env.reset(seed=42)
        total_reward, episode_count = 0, 0
        
        while episode_count < 3:
            normalized_obs = vec_normalize.normalize_obs(obs.reshape(1, -1)).flatten()
            action, _ = model.predict(normalized_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if done:
                print(f"Episode {episode_count + 1} finished with reward: {total_reward}")
                obs, _ = env.reset()
                total_reward = 0
                episode_count += 1

        env.close()
        print(f"Video saved in: {out_dir / 'video'}")

if __name__ == "__main__":
    train_ppo_bipedal(
        total_timesteps=30_000_000,
        n_envs=16,
        render_video=True
    )
