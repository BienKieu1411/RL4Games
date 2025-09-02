import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveOnBestRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if len(rewards) > 0:
                mean_reward = np.mean(rewards)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"New best mean reward: {mean_reward:.2f} - saving model to {self.save_path}")
                    self.model.save(self.save_path)
        return True
