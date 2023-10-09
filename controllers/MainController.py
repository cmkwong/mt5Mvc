from controllers.strategies.StrategyContainer import StrategyContainer
from controllers.myMT5.MT5Controller import MT5Controller
from controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from controllers.PlotController import PlotController
from controllers.ThreadController import ThreadController
from controllers.TimeSeriesController import TimeSeriesController

class MainController:
    def __init__(self):
        self.tg = False
        self.nodeJsApiController = NodeJsApiController()
        self.mt5Controller = MT5Controller(self.nodeJsApiController)
        self.plotController = PlotController()
        self.threadController = ThreadController()
        self.timeSeriesController = TimeSeriesController()
        self.strategyController = StrategyContainer(self.mt5Controller, self.nodeJsApiController)