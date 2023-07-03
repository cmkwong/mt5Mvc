import config
from models.myUtils import timeModel

import os

class Options:
    def __init__(self):
        # getting the updated time string
        self.DT_STRING = timeModel.getTimeS(False, "%Y%m%d%H%M%S")
        # setting the config
        self.general_docs_path = os.path.join(config.PROJECT_PATH, 'docs')
        self.options = {
            'general_docs_path': self.general_docs_path,
            'dt': self.DT_STRING,
            'docs_path': os.path.join(self.general_docs_path, self.DT_STRING),
            'debug': True,
        }
        self.data_options = {
            'start': (2010, 1, 2, 0, 0),
            'end': (2020, 12, 30, 0, 0),
            'symbols': ["EURUSD"],
            'timeframe': '1H',
            'trainTestSplit': 0.7,
            'hist_bins': 100,
            'local_min_path': os.path.join(self.options['docs_path'], "min_data"),
            'local': False,
        }
        self.RL_options = {
            'load_net': False,
            'lr': 0.01,
            'net_folder': '220911104535',  # time that program being run
            'net_file': 'checkpoint-1440000.loader',
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_end': 0.35,
            'gamma': 0.9,
            'reward_steps': 2,
            'net_saved_path': os.path.join(self.options['docs_path'], "net"),
            'val_save_path': os.path.join(self.options['docs_path'], "val"),
            'runs_save_path': os.path.join(*[self.options['general_docs_path'], "runs", self.DT_STRING]),
            'buffer_save_path': os.path.join(self.options['docs_path'], "buffer"),
            'replay_size': 100000,
            'monitor_buffer_size': 10000,
            'replay_start': 10000,  # 10000
            'epsilon_step': 1000000,
            'target_net_sync': 1000,
            'validation_step': 50000,
            'checkpoint_step': 30000,
            'weight_visualize_step': 1000,
            'buffer_monitor_step': 100000,
            'validation_episodes': 5,
            # used in attention network
            'seqLen': 30,
            'netType': 'attention'
        }
        self.state_options = {
            # state options
            'time_cost_pt': 0.05,
            'commission_pt': 8,
            'spread_pt': 15,
            'long_mode': True,
            'reset_on_close': True
        }
        self.env_options = {
            'random_ofs_on_reset': True
        }
        self.tech_params = {
            'ma': [5, 10, 25, 50, 100, 150, 200, 250],
            'bb': [(20, 2, 2, 0), (20, 3, 3, 0), (20, 4, 4, 0), (40, 2, 2, 0), (40, 3, 3, 0), (40, 4, 4, 0)],
            'std': [(5, 1), (20, 1), (50, 1), (100, 1), (150, 1), (250, 1)],
            'rsi': [5, 15, 25, 50, 100, 150, 250],
            'stocOsci': [(5, 3, 3, 0, 0), (14, 3, 3, 0, 0), (21, 14, 14, 0, 0)],
            'macd': [(12, 26, 9), (19, 39, 9)]
        }