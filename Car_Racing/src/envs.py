import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, TransformReward

def make_carracing_env(seed=0, render_mode=None):
    def _init():
        env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=False)
        env = ResizeObservation(env, 84)
        env = GrayScaleObservation(env, keep_dim=True)
        env = TransformReward(env, lambda r: r * 0.1)
        env.reset(seed=seed)
        return env
    return _init
