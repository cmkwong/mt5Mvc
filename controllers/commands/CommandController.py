from models.myUtils.printModel import print_at

# controllers
from controllers.strategies.StrategyContainer import StrategyContainer
from controllers.myMT5.MT5Controller import MT5Controller
from controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from controllers.myStock.StockPriceLoader import StockPriceLoader
from controllers.PlotController import PlotController
from controllers.ThreadController import ThreadController
from controllers.TimeSeriesController import TimeSeriesController

# commands
from controllers.commands.Handler_Control import Handler_Control
from controllers.commands.Handler_Data import Handler_Data
from controllers.commands.Handler_Strategy import Handler_Strategy
from controllers.commands.Handler_Deal import Handler_Deal
from controllers.commands.Handler_Analysis import Handler_Analysis
from controllers.commands.Handler_Test import Handler_Test

from controllers.DfController import DfController


class CommandController:
    def __init__(self):
        self.nodeJsApiController = NodeJsApiController()
        self.plotController = PlotController()
        self.threadController = ThreadController()
        self.timeSeriesController = TimeSeriesController()
        self.stockPriceLoader = StockPriceLoader(self.nodeJsApiController)
        self.mt5Controller = MT5Controller(self.nodeJsApiController)
        self.dfController = DfController(self.plotController)
        self.strategyController = StrategyContainer(self.mt5Controller, self.nodeJsApiController)

        # command handler
        self.handler_control = Handler_Control(self.nodeJsApiController, self.mt5Controller)
        self.handler_data = Handler_Data(self.nodeJsApiController, self.mt5Controller, self.stockPriceLoader, self.dfController)
        self.handler_strategy = Handler_Strategy(self.nodeJsApiController, self.mt5Controller, self.stockPriceLoader, self.threadController, self.strategyController, self.plotController)
        self.handler_deal = Handler_Deal(self.nodeJsApiController, self.mt5Controller, self.stockPriceLoader, self.threadController, self.strategyController, self.plotController)
        self.handler_analysis = Handler_Analysis(self.nodeJsApiController, self.mt5Controller, self.stockPriceLoader, self.threadController, self.strategyController, self.plotController, self.timeSeriesController, self.dfController)
        self.handler_test = Handler_Test(self.nodeJsApiController, self.mt5Controller, self.stockPriceLoader, self.threadController, self.strategyController, self.plotController)

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
