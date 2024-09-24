from mt5Mvc.models.myUtils import paramModel, timeModel
import config
from mt5Mvc.controllers.strategies.Conintegration.Train import Train as Cointegration_Train
from mt5Mvc.controllers.strategies.MovingAverage.Train import Train as MovingAverage_Train
from mt5Mvc.controllers.strategies.MovingAverage.Live import Live as MovingAverage_Live
from mt5Mvc.controllers.strategies.MovingAverage.Backtest import Backtest as MovingAverage_Backtest


class Handler_Strategy:
    def __init__(self, nodeJsApiController, mt5Controller, stockPriceLoader, threadController, strategyController, plotController):
        self.nodeJsApiController = nodeJsApiController
        self.mt5Controller = mt5Controller
        self.stockPriceLoader = stockPriceLoader
        self.threadController = threadController
        self.strategyController = strategyController
        self.plotController = plotController

    def run(self, command):
        if command == '-coinT':
            strategy = Cointegration_Train(self.mt5Controller, self.nodeJsApiController, self.plotController)
            paramFormat = {
                'symbols': [["AUDCAD", "EURUSD", "AUDUSD"], list],
                'start': [(2022, 6, 1, 0, 0), tuple],
                'end': [(2023, 2, 28, 23, 59), tuple],
                'timeframe': ['1H', str],
                "outputPath": ["C:/Users/Chris/projects/221227_mt5Mvc/docs/coin", str]
            }
            param = paramModel.ask_param(paramFormat)
            self.threadController.runThreadFunction(strategy.simpleCheck, **param)

        # finding the best index in moving average
        elif command == '-maT':
            strategy = MovingAverage_Train(self.mt5Controller)
            paramFormat = {
                'symbols': [config.Default_Forex_Symbols, list],
                'timeframe': ['15min', str],
                'start': [(2023, 6, 1, 0, 0), tuple],
                'end': [(2023, 7, 30, 23, 59), tuple],
                'subtest': [False, bool],
            }
            param = paramModel.ask_param(paramFormat)
            strategy.getMaSummaryDf(**param)

        # get the distribution for the specific fast and slow param (the earning distribution)
        elif command == '-mad':
            strategy = MovingAverage_Backtest(self.mt5Controller, self.nodeJsApiController, self.plotController)
            paramFormat = {
                'symbols': [config.Default_Forex_Symbols, list],
                'timeframe': ['15min', str],
                'start': [(2023, 6, 1, 0, 0), tuple],
                'end': [(2023, 7, 30, 23, 59), tuple],
                'fast': [14, int],
                'slow': [22, int],
                'operation': ['long', str],
            }
            param = paramModel.ask_param(paramFormat)
            strategy.getMaDistImg(**param)

        # Moving Average distribution (get params from SQL)
        elif command == '-mads':
            curTime = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
            # get strategy param from
            paramFormat = {
                'strategy_name': ['ma', str],
                'live': [1, int],
                'backtest': [1, int]
            }
            param = paramModel.ask_param(paramFormat)
            params = self.nodeJsApiController.getStrategyParam(**param)
            # run the params
            for i, p in params.iterrows():
                defaultParams = {
                    'curTime': curTime,
                    'symbol': p.symbol,
                    'timeframe': p.timeframe,
                    'start': timeModel.getTimeT(p.start, "%Y-%m-%d %H:%M"),
                    'end': timeModel.getTimeT(p.end, "%Y-%m-%d %H:%M"),
                    'fast': p.fast,
                    'slow': p.slow,
                    'operation': p.operation
                }
                strategy = MovingAverage_Backtest(self.mt5Controller, self.nodeJsApiController, self.plotController)
                strategy.getMaDistImg(**defaultParams)

        # Moving Average Live
        elif command == '-maL':
            strategy = MovingAverage_Live(self.mt5Controller, self.nodeJsApiController)
            strategy.run()

        # Moving Average Live (get params from SQL)
        elif command == '-maLs':
            # dummy function for running the treading strategy
            def foo(paramDf, position_id=None, price_open=None):
                params = MovingAverage_Live.decodeParams(paramDf)
                # run for each param
                for i, p in params.items():
                    # if existed strategy, will not run again
                    if self.strategyController.exist(p['strategy_id']):
                        continue
                    # define require strategy
                    strategy = MovingAverage_Live(self.mt5Controller, self.nodeJsApiController, **p)
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
                'strategy_name': [strategy_name, str],
                'live': [2, int],
                'backtest': [2, int]
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