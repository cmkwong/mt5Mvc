import os
from models.myUtils import sysModel, timeModel

DT_STRING = timeModel.getTimeS(False, "%Y%m%d%H%M%S")

PARAM_PATH = './'
PARAM_FILE = 'param.txt'
TEMP_PATH = '../../docs/temp'

# relative path for different computer
PROJECT_PATH = sysModel.find_required_path(os.getcwd(), '221227_mt5Mvc')
GENERAL_DOCS_PATH = os.path.join(PROJECT_PATH, 'docs')

Broker = 'RoboForex'
Broker_Time_Between_UTC = 0
TimeZone = 'Etc/UTC'   # Check: set(pytz.all_timezones_set) - (Etc/UTC) # Hongkong
DepositCurrency = 'USD'
TypeFilling = 'ioc'
DefaultSymbols = ['AUDJPY', 'AUDCAD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']

# ---------------------------------------------------------------------------------------------
# Forex Train Option
LEARNING_RATE = 0.01
BATCH_SIZE = 64
TRAIN_TEST_SPLIT = 0.7
EPSILON_START = 1.0
EPSILON_END = 0.35
EPSILON_STEP = 1000000
GAMMA = 0.9
REWARD_STEPS = 2
REPLAY_START = 10000  # 10000
REPLAY_SIZE = 100000
TARGET_NET_SYNC = 1000
CHECKPOINT_STEP = 30000
VALIDATION_STEP = 50000
VALIDATION_EPISODES = 5

# Check Performance
HIST_BINS = 100
MONITOR_BUFFER_SIZE = 10000
WEIGHT_VISUALIZE_STEP = 1000
BUFFER_MONITOR_STEP = 100000

# RL Document Path
NET_SAVED_PATH = os.path.join(GENERAL_DOCS_PATH, 'net')
VAL_SAVED_PATH = os.path.join(GENERAL_DOCS_PATH, 'val')
RUNS_SAVED_PATH = os.path.join(*[GENERAL_DOCS_PATH, 'runs', DT_STRING])

# Loading Net
NET_FOLDER = '220911104535'
NET_FILE = 'checkpoint-1440000.loader'

# For Forex State
TIME_COST_POINT = 0.05
COMMISSION_POINT = 8
SPREAD_POINT = 15
RESET_ON_CLOSE = True
RANDOM_OFFSET_ON_RESET = True

TECHNICAL_PARAMS = {
    'ma': [5, 10, 25, 50, 100, 150, 200, 250],
    'bb': [(20, 2, 2, 0), (20, 3, 3, 0), (20, 4, 4, 0), (40, 2, 2, 0), (40, 3, 3, 0), (40, 4, 4, 0)],
    'std': [(5, 1), (20, 1), (50, 1), (100, 1), (150, 1), (250, 1)],
    'rsi': [5, 15, 25, 50, 100, 150, 250],
    'stocOsci': [(5, 3, 3, 0, 0), (14, 3, 3, 0, 0), (21, 14, 14, 0, 0)],
    'macd': [(12, 26, 9), (19, 39, 9)]
}
