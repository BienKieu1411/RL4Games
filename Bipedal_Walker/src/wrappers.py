import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        custom_reward = reward

        hull_angle = obs[2]
        balance_bonus = 0.3 * (1.0 - abs(hull_angle))

        progress = obs[0]
        progress_bonus = 0.1 * progress

        energy_penalty = 0.00035 * np.sum(np.square(action))

        custom_reward = custom_reward + balance_bonus + progress_bonus - energy_penalty

        return obs, custom_reward, terminated, truncated, info
