from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.controllers.myStock.StockPriceLoader import StockPriceLoader
from mt5Mvc.controllers.ThreadController import ThreadController
from mt5Mvc.controllers.strategies.StrategyContainer import StrategyContainer
from mt5Mvc.controllers.myNodeJs.NodeJsApiController import NodeJsApiController

from mt5Mvc.models.myUtils import paramModel, timeModel
import config
from mt5Mvc.controllers.strategies.Conintegration.Train import Train as Cointegration_Train
from mt5Mvc.controllers.strategies.MovingAverage.Train import Train as MovingAverage_Train
from mt5Mvc.controllers.strategies.MovingAverage.Live import Live as MovingAverage_Live
from mt5Mvc.controllers.strategies.MovingAverage.Backtest import Backtest as MovingAverage_Backtest


class Handler_Strategy:
    def __init__(self):
        self.nodeJsApiController = NodeJsApiController()
        self.threadController = ThreadController()
        self.strategyController = StrategyContainer()

    def __call__(self):
        return self.strategyController

    def run(self, command):
        if command == '-coinT':
            strategy = Cointegration_Train()
            paramFormat = {
                'symbols': [["AUDCAD", "EURUSD", "AUDUSD"], list, 'field'],
                'start': [(2022, 6, 1, 0, 0), tuple, 'field'],
                'end': [(2023, 2, 28, 23, 59), tuple, 'field'],
                'timeframe': ['1H', str, 'field'],
                "outputPath": ["C:/Users/Chris/projects/221227_mt5Mvc/docs/coin", str, 'field']
            }
            param = paramModel.ask_param(paramFormat)
            self.threadController.runThreadFunction(strategy.simpleCheck, **param)

        # finding the best index in moving average
        elif command == '-maT':
            strategy = MovingAverage_Train()
            paramFormat = {
                'symbols': [config.Default_Forex_Symbols, list, 'field'],
                'timeframe': ['15min', str, 'field'],
                'start': [(2023, 6, 1, 0, 0), tuple, 'field'],
                'end': [(2023, 7, 30, 23, 59), tuple, 'field'],
                'subtest': [False, bool, 'field'],
            }
            param = paramModel.ask_param(paramFormat)
            strategy.getMaSummaryDf(**param)

        # get the distribution for the specific fast and slow param (the earning distribution)
        elif command == '-mad':
            strategy = MovingAverage_Backtest()
            paramFormat = {
                'symbols': [config.Default_Forex_Symbols, list, 'field'],
                'timeframe': ['15min', str, 'field'],
                'start': [(2023, 6, 1, 0, 0), tuple, 'field'],
                'end': [(2023, 7, 30, 23, 59), tuple, 'field'],
                'fast': [14, int, 'field'],
                'slow': [22, int, 'field'],
                'operation': ['long', str, 'field'],
            }
            param = paramModel.ask_param(paramFormat)
            strategy.getForexMaDistImg(**param)

        # Moving Average distribution (get params from SQL)
        elif command == '-mads':
            # get strategy param from
            paramFormat = {
                'strategy_name': ['ma', str, 'field'],
                'live': [1, int, 'field'],
                'backtest': [1, int, 'field']
            }
            param = paramModel.ask_param(paramFormat)
            params = self.nodeJsApiController.getStrategyParam(**param)
            # run the params
            for i, p in params.iterrows():
                defaultParams = {
                    'symbol': p.symbol,
                    'timeframe': p.timeframe,
                    'start': timeModel.getTimeT(p.start, "%Y-%m-%d %H:%M"),
                    'end': timeModel.getTimeT(p.end, "%Y-%m-%d %H:%M"),
                    'fast': p.fast,
                    'slow': p.slow,
                    'operation': p.operation
                }
                strategy = MovingAverage_Backtest()
                strategy.getForexMaDistImg(**defaultParams)

        # Moving Average Live
        elif command == '-maL':
            strategy = MovingAverage_Live()
            strategy.run()

        # Moving Average Live (get params from SQL)
        elif command == '-maLs':
            # dummy function for running the treading strategy
            def foo(paramDf, position_id=None, price_open=None):
                params = MovingAverage_Live.decodeParams(paramDf)
                # run for each param
                for i, param in params.items():
                    # if existed strategy, will not run again
                    if self.strategyController.exist(param['strategy_id']):
                        continue
                    # define require strategy
                    strategy = MovingAverage_Live(**param)
                    # assign position id and setup ExitPrices (for load param)
                    if position_id and price_open:
                        strategy.position_id = position_id
                        strategy.getExitPrices_tp(price_open)
                    # run thread
                    self.threadController.runThreadFunction(strategy.run)
                    # append the strategy
                    self.strategyController.add(strategy)
                    # print running strategy
                    print(strategy)

            strategy_name = 'ma'
            print("Running position parameter ... ")
            # check if running strategy, if so, then run them first
            for paramDf, position_id, price_open in self.strategyController.load_param(strategy_name):
                # running param
                foo(paramDf, position_id, price_open)

            # ask args
            paramFormat = {
                'strategy_name': [strategy_name, str, 'field'],
                'live': [2, int, 'field'],
                'backtest': [2, int, 'field']
            }
            param = paramModel.ask_param(paramFormat)
            # get the parameter from SQL
            paramDf = self.nodeJsApiController.getStrategyParam(**param)
            if paramDf.empty:
                print("No Param Found. ")
                return False
            # running param
            foo(paramDf)

        else:
            return True