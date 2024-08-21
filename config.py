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
Default_Forex_Symbols = ['AUDJPY', 'AUDCAD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']
Default_Stock_Symbols = ['AAPL', 'AMZN', 'META', 'MSFT', 'TSLA']

# sql query
SQLQUERY_FOREX_DIR = 'C:/Users/Chris/projects/220627_forexWebServer/queries/forex'

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

# RL_tf Document Path
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

# ---------------- MT5 Error Code - Description
MT5_ERROR_CODE = {
    10004: {'constant': 'TRADE_RETCODE_REQUOTE', 'description': 'Requote'},
    10006: {'constant': 'TRADE_RETCODE_REJECT', 'description': 'Request rejected'},
    10007: {'constant': 'TRADE_RETCODE_CANCEL', 'description': 'Request canceled by trader'},
    10008: {'constant': 'TRADE_RETCODE_PLACED', 'description': 'Order placed'},
    10009: {'constant': 'TRADE_RETCODE_DONE', 'description': 'Request completed'},
    10010: {'constant': 'TRADE_RETCODE_DONE_PARTIAL', 'description': 'Only part of the request was completed'},
    10011: {'constant': 'TRADE_RETCODE_ERROR', 'description': 'Request processing error'},
    10012: {'constant': 'TRADE_RETCODE_TIMEOUT', 'description': 'Request canceled by timeout'},
    10013: {'constant': 'TRADE_RETCODE_INVALID', 'description': 'Invalid request'},
    10014: {'constant': 'TRADE_RETCODE_INVALID_VOLUME', 'description': 'Invalid volume in the request'},
    10015: {'constant': 'TRADE_RETCODE_INVALID_PRICE', 'description': 'Invalid price in the request'},
    10016: {'constant': 'TRADE_RETCODE_INVALID_STOPS', 'description': 'Invalid stops in the request'},
    10017: {'constant': 'TRADE_RETCODE_TRADE_DISABLED', 'description': 'Trade is disabled'},
    10018: {'constant': 'TRADE_RETCODE_MARKET_CLOSED', 'description': 'Market is closed'},
    10019: {'constant': 'TRADE_RETCODE_NO_MONEY', 'description': 'There is not enough money to complete the request'},
    10020: {'constant': 'TRADE_RETCODE_PRICE_CHANGED', 'description': 'Prices changed'},
    10021: {'constant': 'TRADE_RETCODE_PRICE_OFF', 'description': 'There are no quotes to process the request'},
    10022: {'constant': 'TRADE_RETCODE_INVALID_EXPIRATION', 'description': 'Invalid order expiration date in the request'},
    10023: {'constant': 'TRADE_RETCODE_ORDER_CHANGED', 'description': 'Order state changed'},
    10024: {'constant': 'TRADE_RETCODE_TOO_MANY_REQUESTS', 'description': 'Too frequent requests'},
    10025: {'constant': 'TRADE_RETCODE_NO_CHANGES', 'description': 'No changes in request'},
    10026: {'constant': 'TRADE_RETCODE_SERVER_DISABLES_AT', 'description': 'Autotrading disabled by server'},
    10027: {'constant': 'TRADE_RETCODE_CLIENT_DISABLES_AT', 'description': 'Autotrading disabled by client terminal'},
    10028: {'constant': 'TRADE_RETCODE_LOCKED', 'description': 'Request locked for processing'},
    10029: {'constant': 'TRADE_RETCODE_FROZEN', 'description': 'Order or position frozen'},
    10030: {'constant': 'TRADE_RETCODE_INVALID_FILL', 'description': 'Invalid order filling type'},
    10031: {'constant': 'TRADE_RETCODE_CONNECTION', 'description': 'No connection with the trade server'},
    10032: {'constant': 'TRADE_RETCODE_ONLY_REAL', 'description': 'Operation is allowed only for live accounts'},
    10033: {'constant': 'TRADE_RETCODE_LIMIT_ORDERS', 'description': 'The number of pending orders has reached the limit'},
    10034: {'constant': 'TRADE_RETCODE_LIMIT_VOLUME', 'description': 'The volume of orders and positions for the symbol has reached the limit'},
    10035: {'constant': 'TRADE_RETCODE_INVALID_ORDER', 'description': 'Incorrect or prohibited order type'},
    10036: {'constant': 'TRADE_RETCODE_POSITION_CLOSED', 'description': 'Position with the specified POSITION_IDENTIFIER has already been closed'},
    10038: {'constant': 'TRADE_RETCODE_INVALID_CLOSE_VOLUME', 'description': 'A close volume exceeds the current position volume'},
    10039: {'constant': 'TRADE_RETCODE_CLOSE_ORDER_EXIST', 'description': 'A close order already exists for a specified position.'},
    10040: {'constant': 'TRADE_RETCODE_LIMIT_POSITIONS',
            'description': 'The number of open positions simultaneously present on an account can be limited by the server settings. After a limit is reached, the server returns the TRADE_RETCODE_LIMIT_POSITIONS error when attempting to place an order. '},
    10041: {'constant': 'TRADE_RETCODE_REJECT_CANCEL', 'description': 'The pending order activation request is rejected, the order is canceled'},
    10042: {'constant': 'TRADE_RETCODE_LONG_ONLY', 'description': 'The request is rejected, because the "Only long positions are allowed" rule is set for the symbol (POSITION_TYPE_BUY)'},
    10043: {'constant': 'TRADE_RETCODE_SHORT_ONLY', 'description': 'The request is rejected, because the "Only short positions are allowed" rule is set for the symbol(POSITION_TYPE_SELL)'},
    10044: {'constant': 'TRADE_RETCODE_CLOSE_ONLY', 'description': 'The request is rejected, because the "Only position closing is allowed" rule is set for the symbol'},
    10045: {'constant': 'TRADE_RETCODE_FIFO_CLOSE', 'description': 'The request is rejected, because "Position closing is allowed only by FIFO rule" flag is set for the trading account (ACCOUNT_FIFO_CLOSE=true)'},
    10046: {'constant': 'TRADE_RETCODE_HEDGE_PROHIBITED', 'description': 'The request is rejected, because the "Opposite positions on a single symbol are disabled" rule is set for the trading account.'},
}

# https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_reason
MT5_DEAL_REASON_CODE = {
    0: {'constant': 'DEAL_REASON_CLIENT', 'description': 'The deal was executed as a result of activation of an order placed from a desktop terminal'},
    1: {'constant': 'DEAL_REASON_MOBILE', 'description': 'The deal was executed as a result of activation of an order placed from a mobile application'},
    2: {'constant': 'DEAL_REASON_WEB', 'description': 'The deal was executed as a result of activation of an order placed from the web platform'},
    3: {'constant': 'DEAL_REASON_EXPERT', 'description': 'The deal was executed as a result of activation of an order placed from an MQL5 program, i.e. an Expert Advisor or a script'},
    4: {'constant': 'DEAL_REASON_SL', 'description': 'The deal was executed as a result of Stop Loss activation'},
    5: {'constant': 'DEAL_REASON_TP', 'description': 'The deal was executed as a result of Take Profit activation'},
    6: {'constant': 'DEAL_REASON_SO', 'description': 'The deal was executed as a result of the Stop Out event'},
    7: {'constant': 'DEAL_REASON_ROLLOVER', 'description': 'The deal was executed due to a rollover'},
    8: {'constant': 'DEAL_REASON_VMARGIN', 'description': 'The deal was executed after charging the variation margin'},
    9: {'constant': 'DEAL_REASON_SPLIT', 'description': 'The deal was executed after the split (price reduction) of an instrument, which had an open position during split announcement'}
}

MT5_DEAL_TYPE = {
    0: {'constant': 'DEAL_TYPE_BUY', 'description': 'Buy'},
    1: {'constant': 'DEAL_TYPE_SELL', 'description': 'Sell'},
    2: {'constant': 'DEAL_TYPE_BALANCE', 'description': 'Balance'},
    3: {'constant': 'DEAL_TYPE_CREDIT', 'description': 'Credit'},
    4: {'constant': 'DEAL_TYPE_CHARGE', 'description': 'Additional charge'},
    5: {'constant': 'DEAL_TYPE_CORRECTION', 'description': 'Correction'},
    6: {'constant': 'DEAL_TYPE_BONUS', 'description': 'Bonus'},
    7: {'constant': 'DEAL_TYPE_COMMISSION', 'description': 'Additional commission'},
    8: {'constant': 'DEAL_TYPE_COMMISSION_DAILY', 'description': 'Daily commission'},
    9: {'constant': 'DEAL_TYPE_COMMISSION_MONTHLY', 'description': 'Monthly commission'},
    10: {'constant': 'DEAL_TYPE_COMMISSION_AGENT_DAILY', 'description': 'Daily agent commission'},
    11: {'constant': 'DEAL_TYPE_COMMISSION_AGENT_MONTHLY', 'description': 'Monthly agent commission'},
    12: {'constant': 'DEAL_TYPE_INTEREST', 'description': 'Interest rate'},
    13: {'constant': 'DEAL_TYPE_BUY_CANCELED', 'description': 'Canceled buy deal. There can be a situation when a previously executed buy deal is canceled.'},
    14: {'constant': 'DEAL_TYPE_SELL_CANCELED', 'description': 'Canceled sell deal. There can be a situation when a previously executed sell deal is canceled. '},
    15: {'constant': 'DEAL_DIVIDEND', 'description': 'Dividend operations'},
    16: {'constant': 'DEAL_DIVIDEND_FRANKED', 'description': 'Franked (non-taxable) dividend operations'},
    17: {'constant': 'DEAL_TAX', 'description': 'Tax charges'},
}

# https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_reason
MT5_ORDER_REASON_CODE = {
    0: {'constant': 'ORDER_REASON_CLIENT', 'description': 'The order was placed from a desktop terminal'},
    1: {'constant': 'ORDER_REASON_MOBILE', 'description': 'The order was placed from a mobile application'},
    2: {'constant': 'ORDER_REASON_WEB', 'description': 'The order was placed from a web platform'},
    3: {'constant': 'ORDER_REASON_EXPERT', 'description': 'The order was placed from an MQL5-program, i.e. by an Expert Advisor or a script'},
    4: {'constant': 'ORDER_REASON_SL', 'description': 'The order was placed as a result of Stop Loss activation'},
    5: {'constant': 'ORDER_REASON_TP', 'description': 'The order was placed as a result of Take Profit activation'},
    6: {'constant': 'ORDER_REASON_SO', 'description': 'The order was placed as a result of the Stop Out event'}
}

MT5_ORDER_TYPE = {
    0: {'constant': 'ORDER_TYPE_BUY', 'description': 'Market Buy order'},
    1: {'constant': 'ORDER_TYPE_SELL', 'description': 'Market Sell order'},
    2: {'constant': 'ORDER_TYPE_BUY_LIMIT', 'description': 'Buy Limit pending order'},
    3: {'constant': 'ORDER_TYPE_SELL_LIMIT', 'description': 'Sell Limit pending order'},
    4: {'constant': 'ORDER_TYPE_BUY_STOP', 'description': 'Buy Stop pending order'},
    5: {'constant': 'ORDER_TYPE_SELL_STOP', 'description': 'Sell Stop pending order'},
    6: {'constant': 'ORDER_TYPE_BUY_STOP_LIMIT', 'description': 'Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price'},
    7: {'constant': 'ORDER_TYPE_SELL_STOP_LIMIT', 'description': 'Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price'},
    8: {'constant': 'ORDER_TYPE_CLOSE_BY', 'description': 'Order to close a position by an opposite one'},
}