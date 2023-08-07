from controllers.myMT5.MT5Controller import MT5Controller
from controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from controllers.PlotController import PlotController
from controllers.StrategyController import StrategyController

class MainController:
    def __init__(self):
        self.tg = False
        self.nodeJsApiController = NodeJsApiController()
        self.mt5Controller = MT5Controller(self.nodeJsApiController)
        self.plotController = PlotController()
        self.strategyController = StrategyController()
