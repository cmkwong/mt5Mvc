import sys
from collections import deque
import time
import numpy as np

class Tracker:
    def __init__(self, writer, rewardMovAver=100, lossMovAver=100):
        self.writer = writer

        # loss
        self.total_losses = deque()
        self.lossMovAver = lossMovAver

        # reward
        self.total_doneRewards, self.total_doneSteps = deque(), deque()
        self.reward_buf, self.steps_buf = [], [] # discard
        self.rewardMovAver = rewardMovAver

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        return self

    def __exit__(self, *args):
        self.writer.close()

    def loss(self, loss, step_idx):
        # https://pythontic.com/containers/deque/introduction
        # append the total loss in list
        self.total_losses.appendleft(loss)
        mean_loss = np.mean(self.total_losses)

        # add the scaler into summary writer
        self.writer.add_scalar(f"loss_{self.lossMovAver}", mean_loss, step_idx)
        self.writer.add_scalar("loss", loss, step_idx)

        # print
        if step_idx % 200 == 0:
            print(f"{step_idx}: {loss}")

        # keep the size fixed within moving average
        if len(self.total_losses) >= self.lossMovAver:
            self.total_losses.pop()

    def getSpeed(self, step_idx):
        # taking last finished time to calculate
        speed = (step_idx - self.ts_frame) / (time.time() - self.ts)
        # update the last step
        self.ts_frame = step_idx
        self.ts = time.time()
        return speed

    def reward(self, doneRewards_doneSteps, step_idx, epsilon=None):
        doneReward, doneSteps = doneRewards_doneSteps
        self.total_doneRewards.appendleft(doneReward)
        self.total_doneSteps.appendleft(doneSteps)
        mean_doneRewards = np.mean(self.total_doneRewards)
        mean_doneSteps = np.mean(self.total_doneSteps)

        # calculate the speed
        speed = self.getSpeed(step_idx)

        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            step_idx, len(self.total_doneRewards) * self.rewardMovAver, mean_doneRewards, mean_doneSteps, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, step_idx)
        self.writer.add_scalar("speed", speed, step_idx)
        self.writer.add_scalar(f"reward_{self.rewardMovAver}", mean_doneRewards, step_idx)
        self.writer.add_scalar("reward", doneReward, step_idx)
        self.writer.add_scalar(f"steps_{self.rewardMovAver}", mean_doneSteps, step_idx)
        self.writer.add_scalar("steps", doneSteps, step_idx)

        # keep the size fixed within moving average
        if len(self.total_doneRewards) >= self.rewardMovAver:
            self.total_doneRewards.pop()
            self.total_doneSteps.pop()

    def reward__DISCARD(self, reward_steps, step_idx, epsilon=None):
        doneReward, doneSteps = reward_steps
        self.reward_buf.append(doneReward)
        self.steps_buf.append(doneSteps)
        if len(self.reward_buf) < self.rewardMovAver:
            return False
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)
        self.reward_buf.clear()
        self.steps_buf.clear()
        self.total_doneRewards.append(reward)
        self.total_doneSteps.append(steps)
        speed = (step_idx - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = step_idx
        self.ts = time.time()
        mean_reward = np.mean(self.total_doneRewards[-100:])
        mean_steps = np.mean(self.total_doneSteps[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            step_idx, len(self.total_doneRewards) * self.rewardMovAver, mean_reward, mean_steps, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, step_idx)
        self.writer.add_scalar("speed", speed, step_idx)
        self.writer.add_scalar("reward_100", mean_reward, step_idx)
        self.writer.add_scalar("reward", reward, step_idx)
        self.writer.add_scalar("steps_100", mean_steps, step_idx)
        self.writer.add_scalar("steps", steps, step_idx)
        if mean_reward > self.rewardTarget:
            print("Solved in %d frames!" % step_idx)
            return True
        return False