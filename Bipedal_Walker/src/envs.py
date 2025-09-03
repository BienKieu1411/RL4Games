import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from .wrappers import CustomRewardWrapper

def make_curriculum_env(render_mode=None, seed=0, difficulty=0):
    def _init():
        if difficulty == 0:
            env = gym.make("BipedalWalker-v3", render_mode=render_mode)
        else:
            env = gym.make("BipedalWalkerHardcore-v3", render_mode=render_mode)

        env.reset(seed=seed)
        env = CustomRewardWrapper(env)
        return env
    return _init

def make_vec_envs(n_envs=1, seed=0, render_mode=None, difficulty=0):
    env_fns = [make_curriculum_env(render_mode=render_mode, seed=seed + i, difficulty=difficulty) for i in range(n_envs)]

    envs = DummyVecEnv(env_fns)
    envs = VecMonitor(envs)
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    return envs
