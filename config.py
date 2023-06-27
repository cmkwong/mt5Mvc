import os
from models.myUtils import sysModel

PARAM_PATH = './'
PARAM_FILE = 'param.txt'

# relative path for different computer
PROJECT_PATH = sysModel.find_required_path(os.getcwd(), '221227_mt5Mvc')

Broker = 'RoboForex'
Broker_Time_Between_UTC = 2
DefaultSymbols = ['AUDJPY', 'AUDCAD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']

# For Forex State
TimeCostPt = 0.05
CommisionPt = 8
SpreadPt = 15
TechicalParams = {
    'ma': [5, 10, 25, 50, 100, 150, 200, 250],
    'bb': [(20, 2, 2, 0), (20, 3, 3, 0), (20, 4, 4, 0), (40, 2, 2, 0), (40, 3, 3, 0), (40, 4, 4, 0)],
    'std': [(5, 1), (20, 1), (50, 1), (100, 1), (150, 1), (250, 1)],
    'rsi': [5, 15, 25, 50, 100, 150, 250],
    'stocOsci': [(5, 3, 3, 0, 0), (14, 3, 3, 0, 0), (21, 14, 14, 0, 0)],
    'macd': [(12, 26, 9), (19, 39, 9)]
}
