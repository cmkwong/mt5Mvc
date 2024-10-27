from mt5Mvc.models.myUtils.printModel import print_at

# controllers
from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.controllers.myStock.StockPriceLoader import StockPriceLoader

# commands
from mt5Mvc.controllers.commands.Handler_Control import Handler_Control
from mt5Mvc.controllers.commands.Handler_Data import Handler_Data
from mt5Mvc.controllers.commands.Handler_Strategy import Handler_Strategy
from mt5Mvc.controllers.commands.Handler_Deal import Handler_Deal
from mt5Mvc.controllers.commands.Handler_Analysis import Handler_Analysis
from mt5Mvc.controllers.commands.Handler_Test import Handler_Test

from mt5Mvc.controllers.DfController import DfController


class CommandController:
    def __init__(self):
        self.stockPriceLoader = StockPriceLoader()
        self.mt5Controller = MT5Controller()

        # command handler
        self.handler_control = Handler_Control(self.mt5Controller, self.stockPriceLoader)
        self.handler_data = Handler_Data()
        self.handler_strategy = Handler_Strategy()
        self.handler_deal = Handler_Deal(self.handler_strategy())
        self.handler_analysis = Handler_Analysis()
        self.handler_test = Handler_Test()

    def run(self, command):
        # if not hit the command, return False
        if not self.handler_control.run(command): return True
        if not self.handler_data.run(command): return True
        if not self.handler_strategy.run(command): return True
        if not self.handler_deal.run(command): return True
        if not self.handler_analysis.run(command): return True
        if not self.handler_test.run(command): return True

        print_at('No command detected. Please input again. ')
        return False

"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
"""
