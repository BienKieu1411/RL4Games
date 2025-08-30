import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from .utils import ensure_dir

def evaluate_and_record(out_dir):
    video_path = out_dir / "video"
    ensure_dir(video_path)

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(env, str(video_path), episode_trigger=lambda x: True)

    best_model_path = out_dir / "best_model" / "best_model.zip"
    model = PPO.load(str(best_model_path))

    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
