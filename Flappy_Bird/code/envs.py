import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

def make_flappy_env(render_mode=None, seed=0):
    def _init():
        env = gym.make("FlappyBird-v0", render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init

def make_vec_envs(n_envs=1, seed=0):
    envs = DummyVecEnv([make_flappy_env(seed=i+seed) for i in range(n_envs)])
    envs = VecMonitor(envs)
    return envs