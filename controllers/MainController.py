from controllers.myMT5.MT5Controller import MT5Controller
from controllers.myNodeJs.ServerController import ServerController as NodeJsServerController

from controllers.StrategyController import StrategyController

class MainController:
    def __init__(self, timezone='Hongkong', deposit_currency='USD', type_filling='ioc'):
        self.tg = False
        self.defaultSymbols = ['AUDJPY', 'AUDCAD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']
        self.mt5Controller = MT5Controller(timezone=timezone, deposit_currency=deposit_currency, type_filling=type_filling)
        self.nodeJsServerController = NodeJsServerController(self.mt5Controller)
        self.strategyController = StrategyController()

