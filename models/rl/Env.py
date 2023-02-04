import numpy as np
import pandas as pd

from models.rl.State import State

from myBacktest import techModel
from myUtils import dicModel


class Env:
    def __init__(self, state, random_ofs_on_reset):
        self.state = state
        self.random_ofs_on_reset = random_ofs_on_reset

    def get_obs_len(self):
        obs = self.reset()
        return len(obs)

    def get_action_space_size(self):
        return self.state.action_space_size

    def reset(self):
        # offset to be zero
        if not self.random_ofs_on_reset:
            self.state.reset(0)
        else:
            # random offset
            self.state.reset(-1)
        obs = self.state.encode()
        return obs

    def step(self, action):
        reward, done = self.state.step(action)
        obs = self.state.encode()
        return obs, reward, done

