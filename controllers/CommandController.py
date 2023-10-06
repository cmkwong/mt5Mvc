import pandas as pd
from pyts.image import GramianAngularField

from models.myUtils.printModel import print_at
from models.myUtils import paramModel, timeModel, printModel
from controllers.strategies.Dealer import Dealer
import config

# Strategy
from controllers.strategies.SwingScalping.Live import Live as SwingScalping_Live
from controllers.strategies.Conintegration.Train import Train as Cointegration_Train
from controllers.strategies.RL_Simple.Train import Train as RL_Simple_Train
from controllers.strategies.MovingAverage.Train import Train as MovingAverage_Train
from controllers.strategies.MovingAverage.Live import Live as MovingAverage_Live
from controllers.strategies.MovingAverage.Backtest import Backtest as MovingAverage_Backtest

from controllers.DfController import DfController
import paramStorage


class CommandController:
    def __init__(self, mainController):
        self.mainController = mainController
        # storing the running strategy
        self.RunningStrategies = []

    def run(self, command):
        # ----------------------- Control -----------------------
        # switch the nodeJS server env: prod / dev
        if command == '-prod' or command == '-dev':
            self.mainController.nodeJsApiController.switchEnv(command[1:])

        # switch the price loader source from mt5 / local
        elif command == '-mt5' or command == '-sql':
            self.mainController.mt5Controller.pricesLoader.switchSource(command[1:])

        # upload the data into mySql server
        elif command == '-upload':
            # setup the source into mt5
            originalSource = self.mainController.mt5Controller.pricesLoader.source
            self.mainController.mt5Controller.pricesLoader.source = 'mt5'
            # upload Prices
            kwargs = {'symbols': config.DefaultSymbols,
                     'start': (2023, 6, 1, 0, 0), 'end': (2023, 6, 30, 23, 59),
                     'timeframe': '1min',
                     'count': 0,
                     'ohlcvs': '111111'}
            obj, kwargs = paramModel.ask_params(self.mainController.mt5Controller.pricesLoader.getPrices, **kwargs)
            Prices = obj(**kwargs)
            self.mainController.nodeJsApiController.uploadOneMinuteForexData(Prices)
            # resume to original source
            self.mainController.mt5Controller.pricesLoader.source = originalSource

        # all symbol info upload
        elif command == '-symbol':
            # upload all_symbol_info
            all_symbol_info = self.mainController.mt5Controller.pricesLoader.all_symbols_info
            kwargs = {'broker': config.Broker}
            obj, kwargs = paramModel.ask_params(self.mainController.nodeJsController.apiController.uploadAllSymbolInfo, **kwargs)
            # append param
            kwargs['all_symbol_info'] = all_symbol_info
            obj(**kwargs)

        # close all deals
        elif command == '-close':
            kwargs = {
                # "position_id": 0,
                # "percent": 0.9,
                # "comment": "Manuel Close"
            }
            obj, kwargs = paramModel.ask_params(self.mainController.mt5Controller.executor.close_request_format, **kwargs)
            request = obj(**kwargs)
            self.mainController.mt5Controller.executor.request_execute(request)

        # close all deals from opened by strategy
        elif command == '-closeAll_s':
            for strategy in self.RunningStrategies:
                strategy.closeDeal('Force to Close')

        # ----------------------- Interface / Display -----------------------
        elif command == '-positions':
            positionsDf = self.mainController.mt5Controller.get_active_positions()
            nextTargetDf = self.mainController.nodeJsApiController.executeMySqlQuery('query_positions_next_target')
            # merge the df
            if not nextTargetDf.empty:
                positionsDf = pd.merge(positionsDf, nextTargetDf, left_on='ticket', right_on='position_id', how='left', right_index=False).fillna('')
            printModel.print_df(positionsDf)

        elif command == '-deals':
            deals = self.mainController.mt5Controller.get_historical_deals(lastDays=1)
            printModel.print_df(deals)

        # ----------------------- Strategy -----------------------
        # load the processing strategy parameter from database
        elif command == '-loads':
            positionsDf = self.mainController.mt5Controller.get_active_positions()
            for i, row in positionsDf.iterrows():
                position_id = row['ticket']
                price_open = row['price_open']
                url = self.mainController.nodeJsApiController.strategyParamUrl
                param = {
                    'position_id': position_id
                }
                paramDf = self.mainController.nodeJsApiController.getDataframe(url, param)
                if paramDf.empty:
                    print(f"{position_id} has no param found. ")
                    continue
                params = MovingAverage_Live.decodeParams(paramDf)
                # run for each param
                for i, p in params.items():
                    # define require strategy
                    strategy = MovingAverage_Live(self.mainController, **p)
                    # assign position id and setup ExitPrices
                    strategy.position_id = position_id
                    strategy.getExitPrices_tp(price_open)
                    # run thread
                    self.mainController.threadController.runThreadFunction(strategy.run)
                    # append the strategy
                    self.RunningStrategies.append(strategy)
                    print(f"{position_id} param found. ")

        # running SwingScalping_Live with all params
        # elif command == '-swL':
        #     defaultParams = paramStorage.METHOD_PARAMS['SwingScalping_Live']
        #     for defaultParam in defaultParams:
        #         strategy = SwingScalping_Live(self.mainController, auto=True)
        #         self.mainController.threadController.runThreadFunction(strategy.run, **defaultParam)
                # self.mainController.strategyController.appendRunning(dicModel.dic2Txt_k(defaultParam), strategy)

        elif command == '-coinT':
            strategy = Cointegration_Train(self.mainController)
            kwargs = {
                'symbols': ["AUDCAD", "EURUSD", "AUDUSD"],
                'start': (2022, 6, 1, 0, 0), 'end': (2023, 2, 28, 23, 59),
                'timeframe': '1H',
                "outputPath": "C:/Users/Chris/projects/221227_mt5Mvc/docs/coin"
            }
            obj, kwargs = paramModel.ask_params(strategy.simpleCheck, **kwargs)
            self.mainController.threadController.runThreadFunction(obj, **kwargs)

        elif command == '-rlT':
            strategy = RL_Simple_Train(self.mainController)
            strategy.run()

        # finding the best index in moving average
        elif command == '-maT':
            strategy = MovingAverage_Train(self.mainController)
            obj, kwargs = paramModel.ask_params(strategy.getMaSummaryDf)
            obj(**kwargs)

        # get the distribution for the specific fast and slow param (the earning distribution)
        elif command == '-mad':
            strategy = MovingAverage_Backtest(self.mainController)
            obj, kwargs = paramModel.ask_params(strategy.getMaDistImg)
            obj(**kwargs)

        # Moving Average distribution (get params from SQL)
        elif command == '-mads':
            curTime = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
            # get strategy param from
            obj, kwargs = paramModel.ask_params(self.mainController.nodeJsApiController.getStrategyParam)
            params = obj(**kwargs)
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
                strategy = MovingAverage_Backtest(self.mainController)
                strategy.getMaDistImg(**defaultParams)

        # Moving Average Live
        elif command == '-maL':
            strategy = MovingAverage_Live(self.mainController)
            obj, kwargs = paramModel.ask_params(strategy.run)
            self.mainController.threadController.runThreadFunction(obj, **kwargs)
            # strategy.run(**defaultParam)

        # Moving Average Live (get params from SQL)
        elif command == '-maLs':
            # ask args
            kwargs = {'strategy_name': 'ma',
                      'live': 2,
                      'backtest': 2
                      }
            obj, kwargs = paramModel.ask_params(self.mainController.nodeJsApiController.getStrategyParam, **kwargs)
            # get the parameter from SQL
            paramDf = obj(**kwargs)
            if paramDf.empty:
                print("No Param Found. ")
                return False
            params = MovingAverage_Live.decodeParams(paramDf)
            # run for each param
            for i, p in params.items():
                # define require strategy
                strategy = MovingAverage_Live(self.mainController, **p)
                # run thread
                self.mainController.threadController.runThreadFunction(strategy.run)
                # append the strategy
                self.RunningStrategies.append(strategy)

        # ----------------------- Analysis -----------------------
        # running Covariance_Live with all params
        elif command == '-cov':
            kwargs = {
                'symbols': config.DefaultSymbols,
                'start': (2022, 10, 30, 0, 0),
                'end': (2022, 12, 16, 21, 59),
                'timeframe': '1H'
            }
            obj, kwargs = paramModel.ask_params(self.mainController.mt5Controller.pricesLoader.getPrices, **kwargs)
            Prices = obj(**kwargs)
            # get the correlation table
            corelaDf = self.mainController.timeSeriesController.get_corelaDf(Prices.cc)
            corelaTxtDf = pd.DataFrame()
            for symbol in kwargs['symbols']:
                series = corelaDf[symbol].sort_values(ascending=False).drop(symbol)
                corelaDict = dict(series)
                corelaTxtDf[symbol] = pd.Series([f"{key}: {value * 100:.2f}%" for key, value in corelaDict.items()])
            # print the correlation tables
            printModel.print_df(corelaDf)
            # print the highest correlated symbol tables
            printModel.print_df(corelaTxtDf)

        # view the time series into Gramian Angular Field Image
        elif command == '-gaf':
            obj, kwargs = paramModel.ask_params(self.mainController.mt5Controller.pricesLoader.getPrices)
            Prices = obj(**kwargs)
            ohlcvs_dfs = Prices.getOhlcvsFromPrices()
            for symbol, nextTargetDf in ohlcvs_dfs.items():
                gasf = GramianAngularField(method='summation', image_size=1.0)
                X_gasf = gasf.fit_transform(nextTargetDf['close'].values.reshape(1, -1))
                gadf = GramianAngularField(method='difference', image_size=1.0)
                X_gadf = gadf.fit_transform(nextTargetDf['close'].values.reshape(1, -1))
                self.mainController.plotController.getGafImg(X_gasf[0, :, :], X_gadf[0, :, :], nextTargetDf['close'], f"{symbol}_gaf.jpg")
                print(f"{symbol} gaf generated. ")

        elif command == '-corelate':
            pass

        # get the summary of df
        elif command == '-dfsumr':
            dfController = DfController()
            # read the df
            print("Read File csv/exsl: ")
            obj_readParam, readParam = paramModel.ask_params(DfController.readAsDf)
            print("Output pdf: ")
            obj_sumParam, sumParam = paramModel.ask_params(DfController.summaryPdf)
            nextTargetDf = obj_readParam(**readParam)
            obj_sumParam(nextTargetDf, **sumParam)

        # ----------------------- Testing -----------------------
        # testing for getting the data from sql / mt5 by switch the data source
        elif command == '-testPeriod':
            self.mainController.mt5Controller.pricesLoader.getPrices(
                symbols=['USDJPY'],
                start=(2023, 2, 18, 0, 0),
                end=(2023, 7, 20, 0, 0),
                timeframe='1min'
            )
        # testing for getting current data from sql / mt5 by switch the data source
        elif command == '-testCurrent':
            self.mainController.mt5Controller.pricesLoader.getPrices(
                symbols=['USDJPY'],
                count=1000,
                timeframe='15min'
            )
        elif command == '-testOrder':
            pass

        elif command == '-testDeal':
            # define the dealer
            dealer = Dealer(self.mainController,
                           strategy_name='Test',
                           strategy_detail='Test_detail',
                           symbol='USDJPY',
                           timeframe='15min',
                           operation='long',
                           lot=0.1,
                           pt_sl=500,
                           exitPoints={900: 0.75, 1200: 0.2, 1500: 0.05}
                           )
            dealer.openDeal()
            dealer.closeDeal()
            print()
        elif command == '-testMt5':
            historicalOrder = self.mainController.mt5Controller.get_historical_order(lastDays=3, position_id=338232986)
            historicalDeals = self.mainController.mt5Controller.get_historical_deals(lastDays=3, position_id=338232986)
            print()
        else:
            print_at('No command detected. Please input again. ')


"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
"""
