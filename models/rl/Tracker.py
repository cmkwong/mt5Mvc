import sys
import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class Tracker:
    def __init__(self, writer, rewardTarget, rewardSize=1, lossSize=1):
        self.writer = writer

        # loss
        self.loss_buf = []
        self.total_loss = []
        self.steps_buf = []
        self.lossSize = lossSize
        self.capacity = lossSize*10

        # reward
        self.rewardTarget = rewardTarget
        self.reward_buf = []
        self.steps_buf = []
        self.rewardSize = rewardSize

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_steps = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def loss(self, loss, frame):
        # assert (isinstance(loss, np.float))
        self.loss_buf.append(loss)
        if len(self.loss_buf) < self.lossSize:
            return False
        mean_loss = np.mean(self.loss_buf)
        self.loss_buf.clear()
        self.total_loss.append(mean_loss)
        movingAverage_loss = np.mean(self.total_loss[-100:])
        if len(self.total_loss) > self.capacity:
            self.total_loss = self.total_loss[1:]

        self.writer.add_scalar("loss_100", movingAverage_loss, frame)
        self.writer.add_scalar("loss", mean_loss, frame)

    def reward(self, reward_steps, frame, epsilon=None):
        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        if len(self.reward_buf) < self.rewardSize:
            return False
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)
        self.reward_buf.clear()
        self.steps_buf.clear()
        self.total_rewards.append(reward)
        self.total_steps.append(steps)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards)*self.rewardSize, mean_reward, mean_steps, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("steps_100", mean_steps, frame)
        self.writer.add_scalar("steps", steps, frame)
        if mean_reward > self.rewardTarget:
            print("Solved in %d frames!" % frame)
            return True
        return False