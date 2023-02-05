import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from controllers.strategies.rl.Options import Options
from myDataFeed.myMt5.MT5Controller import MT5Controller
from models.rl.State import State
from models.rl.Env import Env
from models.rl.Nets import AttentionTimeSeries, SimpleFFDQN, SimpleLSTM
from models.rl.Actions import EpsilonGreedyActionSelector
from models.rl.Agents import DQNAgentAttn
from models.rl.ExperienceSource import ExperienceSourceFirstLast
from models.rl.ExperienceBuffer import ExperienceReplayBuffer
from models.rl.Validator import StockValidator
from models.rl.Tracker import Tracker


class Train(Options):
    def __init__(self):
        super(Train, self).__init__()
        self.initTrainTestSet()
        self.initState()
        self.initEnv()
        self.initAgent()
        self.initExpSource()
        self.initExpBuffer()
        self.initOptimizer()
        self.initValidator()
        self.initSummaryWriter()
        self.initTracker()
        self.step_idx = 0

    def initTrainTestSet(self):
        self.mt5Controller = MT5Controller(timezone=self.data_options['timezone'], deposit_currency=self.data_options['deposit_currency'])
        # get the loader
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=self.data_options['symbols'],
                                                           start=self.data_options['start'],
                                                           end=self.data_options['end'],
                                                           timeframe=self.data_options['timeframe']
                                                           )
        # split into train set and test set
        self.Train_Prices, self.Test_Prices = self.mt5Controller.pricesLoader.split_Prices(Prices, percentage=self.data_options['trainTestSplit'])

    def initState(self):
        # build the state
        self.state = State(self.Train_Prices, self.data_options['symbol'], self.tech_params,
                           self.state_options['time_cost_pt'], self.state_options['commission_pt'], self.state_options['spread_pt'], self.state_options['long_mode'],
                           self.mt5Controller.all_symbol_info, self.state_options['reset_on_close'])
        self.state_val = State(self.Test_Prices, self.data_options['symbol'], self.tech_params,
                               self.state_options['time_cost_pt'], self.state_options['commission_pt'], self.state_options['spread_pt'], self.state_options['long_mode'],
                               self.mt5Controller.all_symbol_info, False)

    def initEnv(self):
        # build the env
        self.env = Env(self.state, self.env_options['random_ofs_on_reset'])
        self.env_val = Env(self.state_val, self.env_options['random_ofs_on_reset'])

    def _initSelector(self):
        self.selector = EpsilonGreedyActionSelector(self.RL_options['epsilon_start'])

    # load net
    def _loadNet(self):
        loadedPath = os.path.join(*[self.general_docs_path, self.RL_options['dt_str'], 'net'])
        with open(os.path.join(*[loadedPath, self.RL_options['net_file']]), "rb") as f:
            checkpoint = torch.load(f)
        # net = AttentionTimeSeries(hiddenSize=128, inputSize=55, seqLen=30, batchSize=128, outputSize=3, statusSize=2, pdrop=0.1)
        self.net.load_state_dict(checkpoint['state_dict'])

    def _initNet(self):
        # check using different net
        if self.RL_options['netType'] == 'simple':
            self.net = SimpleFFDQN(self.env.get_obs_len(), self.env.get_action_space_size())
        elif self.RL_options['netType'] == 'attention':
            self.net = AttentionTimeSeries(hiddenSize=128, inputSize=55, seqLen=30, batchSize=128, outputSize=3, statusSize=2, pdrop=0.1)
        # if loading net
        if (self.RL_options['load_net']):
            self._loadNet()
        self.net.to(torch.device("cuda"))  # pass into gpu

    # init the agent
    def initAgent(self):
        self.agent = DQNAgentAttn(self.net, self.selector)

    # init experience source
    def initExpSource(self):
        self.exp_source = ExperienceSourceFirstLast(self.env, self.agent, self.RL_options['gamma'], steps_count=self.RL_options['reward_steps'])

    # init experience buffer
    def initExpBuffer(self):
        self.buffer = ExperienceReplayBuffer(self.exp_source, self.RL_options['replay_size'])

    # create optimizer
    def initOptimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.RL_options['lr'])

    # running testing environment
    def initValidator(self):
        self.validator = StockValidator(self.env_val, save_path=os.path.join(*[self.RL_options['val_save_path']]), comission=0.1)

    # writer for plotting the graph in tensorboard
    def initSummaryWriter(self):
        self.summaryWriter = SummaryWriter(log_dir=os.path.join(self.RL_options['runs_save_path']), comment="ForexRL")

    # track the loss and reward. Console the log in screen
    def initTracker(self):
        writer = SummaryWriter(log_dir=os.path.join(self.RL_options['runs_save_path']), comment="ForexRL")
        self.tracker = Tracker(writer, rewardMovAver=1, lossMovAver=1)

    def run(self):
        with self.tracker:
            while (True):
                self.agent.switchNetMode('populate')
                self.buffer.populate(1)
                self.selector.epsilon = max(self.RL_options['epsilon_end'], self.RL_options['epsilon_start'] - self.step_idx * 0.75 / self.RL_options['epsilon_step'])

                doneRewards_doneSteps = self.exp_source.pop_rewards_steps()

                if doneRewards_doneSteps:
                    self.tracker.reward(doneRewards_doneSteps, self.step_idx, self.selector.epsilon)
                if len(self.buffer) < self.RL_options['replay_start']:
                    continue

                self.optimizer.zero_grad()
                batch = self.buffer.sample(self.RL_options['batch_size'])

                # init the hidden both in network and tgt network
                self.agent.switchNetMode('train')
                loss_v = self.agent.calc_loss(batch, self.RL_options['gamma'] ** self.RL_options['reward_steps'])
                loss_v.backward()
                self.optimizer.step()
                loss_value = loss_v.item()
                self.tracker.loss(loss_value, self.step_idx)

                # sync the target net with net
                if self.step_idx % self.RL_options['target_net_sync'] == 0:
                    self.agent.sync()

                # save the check point
                if self.step_idx % self.RL_options['checkpoint_step'] == 0:
                    # idx = step_idx // CHECKPOINT_EVERY_STEP
                    checkpoint = {
                        "state_dict": self.agent.net.state_dict()
                    }
                    with open(os.path.join(*[self.RL_options['net_saved_path'], f"checkpoint-{self.step_idx}.loader"]), "wb") as f:
                        torch.save(checkpoint, f)
