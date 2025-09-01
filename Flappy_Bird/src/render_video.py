import gymnasium as gym
from stable_baselines3 import PPO
from gym.wrappers import RecordVideo
from pathlib import Path

def render_model(model_path, video_path, env_name="FlappyBird-v0", seed=0):
    Path(video_path).mkdir(parents=True, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, str(video_path), episode_trigger=lambda x: True)

    model = PPO.load(model_path)

    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

    env.close()