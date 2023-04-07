import os
from models.myUtils import sysModel

PARAM_PATH = './'
PARAM_FILE = 'param.txt'

# relative path for different computer
PROJECT_PATH = sysModel.find_required_path(os.getcwd(), '221227_mt5Mvc')

Broker = 'RoboForex'
Broker_Time_Between_UTC = 2
DefaultSymbols = ['AUDJPY', 'AUDCAD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']