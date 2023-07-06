import os
import torch
from torch.utils.tensorboard import SummaryWriter

import config

from models.AI.ForexState import ForexState, AttnForexState
from models.AI.Env import Env

from models.AI.Nets.Attn import AttentionTimeSeries
from models.AI.Nets.TimeSeries import SimpleLSTM
from models.AI.Nets.FullyConnected import SimpleFFDQN

from models.AI.Actions import EpsilonGreedyActionSelector
from models.AI.Agents import DQNAgentAttn
from models.AI.ExperienceSource import ExperienceSourceFirstLast
from models.AI.ExperienceBuffer import ExperienceReplayBuffer
from models.AI.Validator import StockValidator
from models.AI.Tracker import Tracker


class Train:
    def __init__(self, mainController, *,
                 symbol='USDJPY',
                 timeframe='1H',
                 start=(2018, 1, 2, 0, 0), end=(2020, 12, 30, 0, 0),
                 seq_len=30, net_type='attention',
                 load_net=False,
                 long_mode=False
                 ):
        self.mainController = mainController
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsApiController = mainController.nodeJsApiController
        # self.all_symbol_info = mainController.mt5Controller.mt5PricesLoader.all_symbols_info
        # self.symbol = symbol

        Prices = self.mt5Controller.mt5PricesLoader.getPrices(symbols=[symbol], start=start, end=end, timeframe=timeframe, count=0, ohlcvs='111111')
        # split into train set and test set
        self.Train_Prices, self.Test_Prices = Prices.split_Prices(percentage=config.TRAIN_TEST_SPLIT)

        value = self.Train_Prices.getValueDiff(1, 100)

        # build the state
        if net_type == 'simple':
            self.state = ForexState(self.Train_Prices, config.TECHNICAL_PARAMS, long_mode, config.RESET_ON_CLOSE)
            self.state_val = ForexState(self.Test_Prices, config.TECHNICAL_PARAMS, long_mode, False)
        elif net_type == 'attention':
            # Prices, symbol, tech_params, time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close
            self.state = AttnForexState(self.Train_Prices, config.TECHNICAL_PARAMS, long_mode, config.RESET_ON_CLOSE, seq_len)
            self.state_val = AttnForexState(self.Test_Prices, config.TECHNICAL_PARAMS, long_mode, False, seq_len)

        # build the env
        self.env = Env(self.state, config.RANDOM_OFFSET_ON_RESET)
        self.env_val = Env(self.state_val, False)

        # selector
        self.selector = EpsilonGreedyActionSelector(config.EPSILON_START)

        # check using different net
        if net_type == 'simple':
            self.net = SimpleFFDQN(self.env.get_obs_len(), self.env.get_action_space_size())
        elif net_type == 'attention':
            self.net = AttentionTimeSeries(hiddenSize=128, inputSize=55, seqLen=30, batchSize=128, outputSize=3, statusSize=2, pdrop=0.1)

        # if loading net
        if load_net:
            # load net
            loadedPath = os.path.join(*[config.GENERAL_DOCS_PATH, 'net', config.NET_FOLDER])
            with open(os.path.join(*[loadedPath, config.NET_FILE]), "rb") as f:
                checkpoint = torch.load(f)
            # net = AttentionTimeSeries(hiddenSize=128, inputSize=55, seqLen=30, batchSize=128, outputSize=3, statusSize=2, pdrop=0.1)
            self.net.load_state_dict(checkpoint['state_dict'])

        # pass net into gpu
        self.net.to(torch.device("cuda"))

        # init agent
        self.agent = DQNAgentAttn(self.net, self.selector)

        # init experience source
        self.exp_source = ExperienceSourceFirstLast(self.env, self.agent, config.GAMMA, steps_count=config.REWARD_STEPS)

        # init experience buffer
        self.buffer = ExperienceReplayBuffer(self.exp_source, config.REPLAY_SIZE)

        # create optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.LEARNING_RATE)

        # running testing environment
        self.validator = StockValidator(self.env_val, save_path=config.VAL_SAVED_PATH, comission=0.1)

        # writer for plotting the graph in tensorboard
        self.summaryWriter = SummaryWriter(log_dir=config.RUNS_SAVED_PATH, comment="ForexRL")

        # track the loss and reward. Console the log in screen
        writer = SummaryWriter(log_dir=config.RUNS_SAVED_PATH, comment="ForexRL")
        self.tracker = Tracker(writer, rewardMovAver=1, lossMovAver=1)

        # self.initEnv()
        # self.initNet()
        # self.initSelector()
        # self.initAgent()
        # self.initExpSource()
        # self.initExpBuffer()
        # self.initOptimizer()
        # self.initValidator()
        # self.initSummaryWriter()
        # self.initTracker()
        self.step_idx = 0

    @property
    def getName(self):
        parentFolder = os.path.basename(os.getcwd())
        return f'{parentFolder}({self.__class__.__name__})'

    def run(self):
        with self.tracker:
            while (True):
                self.agent.switchNetMode('populate')
                self.buffer.populate(1)
                self.selector.epsilon = max(config.EPSILON_END, config.EPSILON_START - self.step_idx * 0.75 / config.EPSILON_STEP)

                doneRewards_doneSteps = self.exp_source.pop_rewards_steps()

                if doneRewards_doneSteps:
                    self.tracker.reward(doneRewards_doneSteps, self.step_idx, self.selector.epsilon)
                if len(self.buffer) < config.REPLAY_START:
                    continue

                self.optimizer.zero_grad()
                batch = self.buffer.sample(config.BATCH_SIZE)

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
                if self.step_idx % self.RL_options['checkpoint_step'] == -1:
                    # idx = step_idx // CHECKPOINT_EVERY_STEP
                    checkpoint = {
                        'state_dict': self.agent.net.state_dict()
                    }
                    with open(os.path.join(*[self.RL_options['net_saved_path'], f"checkpoint-{self.step_idx}.loader"]), "wb") as f:
                        torch.save(checkpoint, f)
