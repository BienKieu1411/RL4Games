from stable_baselines3.common.callbacks import BaseCallback

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % 100000 == 0:
            self.model.get_env().save(str(self.save_path))
        return True
