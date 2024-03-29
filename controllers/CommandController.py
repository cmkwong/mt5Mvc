import pandas as pd
import os
from datetime import datetime, timedelta
from pyts.image import GramianAngularField

from models.myUtils.printModel import print_at
from models.myUtils import paramModel, timeModel, printModel, fileModel, inputModel
from controllers.strategies.Dealer import Dealer
import config

# controllers
from controllers.strategies.StrategyContainer import StrategyContainer
from controllers.myMT5.MT5Controller import MT5Controller
from controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from controllers.myStock.StockPriceLoader import StockPriceLoader
from controllers.PlotController import PlotController
from controllers.ThreadController import ThreadController
from controllers.TimeSeriesController import TimeSeriesController

# Strategy
from controllers.strategies.Conintegration.Train import Train as Cointegration_Train
from controllers.strategies.MovingAverage.Train import Train as MovingAverage_Train
from controllers.strategies.MovingAverage.Live import Live as MovingAverage_Live
from controllers.strategies.MovingAverage.Backtest import Backtest as MovingAverage_Backtest

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

    def run(self, command):
        # ----------------------- Control -----------------------
        # switch the nodeJS server env: prod / dev
        if command == '-prod' or command == '-dev':
            self.nodeJsApiController.switchEnv(command[1:])

        # switch the price loader source from mt5 / local
        elif command == '-mt5' or command == '-local':
            self.mt5Controller.pricesLoader.switchSource(command[1:])

        # upload the data into mySql server
        elif command == '-upload':
            # setup the source into mt5
            originalSource = self.mt5Controller.pricesLoader.source
            self.mt5Controller.pricesLoader.source = 'mt5'
            # upload Prices
            paramFormat = {
                    'symbols':      [config.Default_Forex_Symbols, list],
                    'start':        [(2023, 3, 24, 0, 0), tuple],
                    'end':          [(2023, 4, 24, 23, 59), tuple],
                    'timeframe':    ['1min', str],
                    'count':        [0, int],
                    'ohlcvs':       ['111111', str]
            }
            obj, param = paramModel.ask_param_fn(self.mt5Controller.pricesLoader.getPrices, **paramFormat)
            Prices = obj(**param)
            self.nodeJsApiController.uploadOneMinuteForexData(Prices)
            # resume to original source
            self.mt5Controller.pricesLoader.source = originalSource

        # download the stock data
        elif command == '-download_test':
            obj, param = paramModel.ask_param_fn(self.stockPriceLoader.getPrices)
            Prices = obj(**param)
            print()

        # all symbol info upload
        elif command == '-symbol':
            # upload all_symbol_info
            all_symbol_info = self.mt5Controller.symbolController.get_all_symbols_info()
            paramFormat = {'broker': config.Broker}
            obj, param = paramModel.ask_param_fn(self.nodeJsApiController.uploadAllSymbolInfo, **paramFormat)
            # append param
            param['all_symbol_info'] = all_symbol_info
            obj(**param)

        # close all deals
        elif command == '-close':
            paramFormat = {
                # "position_id": 0,
                "percent": [1.0, float],
                "comment": ["Manuel Close", str]
            }
            obj, param = paramModel.ask_param_fn(self.mt5Controller.executor.close_request_format, **paramFormat)
            request = obj(**param)
            self.mt5Controller.executor.request_execute(request)

        # close all deals from opened by strategy
        elif command == '-closeAll_s':
            for id, strategy in self.strategyController:
                # if succeed to close deal
                if strategy.closeDeal(comment='Force to Close'):
                    print(f"Strategy id: {id} closed. ")

        # execute the query and get the dataframe
        elif command == '-sql':
            # get the query name
            fileList = fileModel.getFileList(config.SQLQUERY_FOREX_DIR)
            userInput = inputModel.askSelection(fileList)
            queryName = fileList[userInput].rsplit('.', 1)[0]
            df = self.nodeJsApiController.executeMySqlQuery(queryName)
            # out and open the excel
            fullPath = os.path.join('./docs/excel', f"{queryName}_{timeModel.getTimeS(outputFormat='%Y%m%d%H%M%S')}.xlsx")
            df.to_excel(fullPath)
            os.system(os.path.abspath(fullPath))

        # ----------------------- Interface / Display -----------------------
        elif command == '-positions':
            positionsDf = self.mt5Controller.get_active_positions()
            nextTargetDf = self.nodeJsApiController.executeMySqlQuery('query_positions_next_target')
            # merge the positionsDf and nextTargetDf
            if not nextTargetDf.empty:
                positionsDf['ticket'] = positionsDf['ticket'].astype('str')
                positionsDf = pd.merge(positionsDf, nextTargetDf, left_on='ticket', right_on='position_id', how='left', right_index=False).fillna('')
                # positionsDf['position_id'] = positionsDf['position_id'].astype('Int64').astype('str')
            printModel.print_df(positionsDf)
            # account balance
            self.mt5Controller.print_account_balance()

        # check the deal performance from-to
        elif command == '-performance':
            now = datetime.now()
            dateFormat = "%Y-%m-%d %H:%M:%S"
            paramFormat = {
                'datefrom': ((now + timedelta(hours=-48)).strftime(dateFormat), str),
                'dateto': (now.strftime(dateFormat), str)
            }
            param = paramModel.ask_param(paramFormat)
            df = self.nodeJsApiController.executeMySqlQuery('query_position_performance', param)
            printModel.print_df(df)
            # account balance
            self.mt5Controller.print_account_balance()

        elif command == '-deals':
            deals = self.mt5Controller.get_historical_deals(lastDays=1)
            printModel.print_df(deals)

        # ----------------------- Strategy -----------------------
        elif command == '-coinT':
            strategy = Cointegration_Train(self.mt5Controller, self.nodeJsApiController, self.plotController)
            paramFormat = {
                'symbols': [["AUDCAD", "EURUSD", "AUDUSD"], list],
                'start': [(2022, 6, 1, 0, 0), tuple],
                'end': [(2023, 2, 28, 23, 59), tuple],
                'timeframe': ['1H', str],
                "outputPath": ["C:/Users/Chris/projects/221227_mt5Mvc/docs/coin", str]
            }
            obj, param = paramModel.ask_param_fn(strategy.simpleCheck, **paramFormat)
            self.threadController.runThreadFunction(obj, **param)

        # elif command == '-rlT':
        #     strategy = RL_Simple_Train(self)
        #     strategy.run()

        # finding the best index in moving average
        elif command == '-maT':
            strategy = MovingAverage_Train(self.mt5Controller)
            obj, param = paramModel.ask_param_fn(strategy.getMaSummaryDf)
            obj(**param)

        # get the distribution for the specific fast and slow param (the earning distribution)
        elif command == '-mad':
            strategy = MovingAverage_Backtest(self.mt5Controller, self.nodeJsApiController, self.plotController)
            obj, param = paramModel.ask_param_fn(strategy.getMaDistImg)
            obj(**param)

        # Moving Average distribution (get params from SQL)
        elif command == '-mads':
            curTime = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
            # get strategy param from
            obj, param = paramModel.ask_param_fn(self.nodeJsApiController.getStrategyParam)
            params = obj(**param)
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
            obj, param = paramModel.ask_param_fn(strategy.run)
            self.threadController.runThreadFunction(obj, **param)
            # strategy.run(**defaultParam)

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
            paramFormat = {'strategy_name': [strategy_name, str],
                      'live': [2, int],
                      'backtest': [2, int]
                      }
            obj, param = paramModel.ask_param_fn(self.nodeJsApiController.getStrategyParam, **paramFormat)
            # get the parameter from SQL
            paramDf = obj(**param)
            if paramDf.empty:
                print("No Param Found. ")
                return False
            # running param
            foo(paramDf)

        # ----------------------- Analysis -----------------------

        # ----------------------- Analysis -----------------------
        # seeing the decomposition of time series
        elif command == '-dct':
            symbol = 'USDJPY'
            paramFormat = {
                'symbols': [symbol],
                'start': (2023, 5, 30, 0, 0),
                'end': (2023, 9, 30, 23, 59),
                'timeframe': '15min'
            }
            obj, param = paramModel.ask_param_fn(self.mt5Controller.pricesLoader.getPrices, **paramFormat)
            Prices = obj(**param)
            season, trend, resid = self.timeSeriesController.decompose_timeData(Prices.close)
            axs = self.plotController.getAxes(4, 1, (90, 50))
            self.plotController.plotSimpleLine(axs[0], Prices.close.iloc[:, 0], symbol)
            self.plotController.plotSimpleLine(axs[1], season, 'season')
            self.plotController.plotSimpleLine(axs[2], trend, 'trend')
            self.plotController.plotSimpleLine(axs[3], resid, 'resid')
            # self.plotController.show()
            self.plotController.saveImg('./docs/img')
            print()

        # running Covariance_Live with all params
        elif command == '-cov':
            paramFormat = {
                'symbols': [config.Default_Forex_Symbols, list],
                'start': [(2022, 10, 30, 0, 0), tuple],
                'end': [(2022, 12, 16, 21, 59), tuple],
                'timeframe': ['1H', str]
            }
            obj, param = paramModel.ask_param_fn(self.mt5Controller.pricesLoader.getPrices, **paramFormat)
            Prices = obj(**param)
            # get the correlation table
            corelaDf = self.timeSeriesController.get_corelaDf(Prices.cc)
            corelaTxtDf = pd.DataFrame()
            for symbol in param['symbols']:
                series = corelaDf[symbol].sort_values(ascending=False).drop(symbol)
                corelaDict = dict(series)
                corelaTxtDf[symbol] = pd.Series([f"{key}: {value * 100:.2f}%" for key, value in corelaDict.items()])
            # print the correlation tables
            printModel.print_df(corelaDf)
            # print the highest correlated symbol tables
            printModel.print_df(corelaTxtDf)

        # view the time series into Gramian Angular Field Image (forex / stock)
        elif command == '-gaf':
            ans = inputModel.askSelection(["Forex", "Stock"])
            # get the price
            if ans == 0:
                obj, param = paramModel.ask_param_fn(self.mt5Controller.pricesLoader.getPrices)
            else:
                obj, param = paramModel.ask_param_fn(self.stockPriceLoader.getPrices)
            Prices = obj(**param)
            ohlcvs_dfs = Prices.getOhlcvsFromPrices()
            # get time string
            curTimeStr = timeModel.getTimeS(outputFormat="%Y%m%d-%H%M%S")
            for symbol, nextTargetDf in ohlcvs_dfs.items():
                X_gasf, X_gadf = self.timeSeriesController.getGAF(nextTargetDf['close'])
                self.plotController.getGafImg(X_gasf[0, :, :], X_gadf[0, :, :], nextTargetDf['close'], './docs/img', f"{curTimeStr}_{symbol}_gaf.jpg")

        # get the summary of df
        elif command == '-dfsumr':
            # read the df
            print("Read File csv/exsl: ")
            obj_readParam, readParam = paramModel.ask_param_fn(self.dfController.readAsDf)
            print("Output pdf: ")
            obj_sumParam, sumParam = paramModel.ask_param_fn(self.dfController.summaryPdf)
            _, nextTargetDf = obj_readParam(**readParam)
            obj_sumParam(nextTargetDf, **sumParam)

        # read the price csv and upload into server
        elif command == '-upload_stock':
            params = paramModel.ask_param({
                'path': ['C:/Users/Chris/projects/221227_mt5Mvc/docs/datas/US Stock', str]
            })
            filename, df = self.dfController.readAsDf(**params)
            # ask table name
            params = paramModel.ask_param({
                'tableName': [filename.lower().split('.', -1)[0], str]
            })
            # change the datetime
            df['datetime'] = pd.to_datetime(df['datetime'])

            # change the index into datetime
            if (self.nodeJsApiController.postDataframe(self.nodeJsApiController.uploadTableUrl, df, {'schemaName': 'stock', **params})):
                print(f"{params}: {len(df)} data being uploaded. ")

        # ----------------------- Testing -----------------------
        # testing for getting the data from sql / mt5 by switch the data source
        elif command == '-testPeriod':
            self.mt5Controller.pricesLoader.getPrices(
                symbols=['USDJPY'],
                start=(2023, 2, 18, 0, 0),
                end=(2023, 7, 20, 0, 0),
                timeframe='1min'
            )
        # testing for getting current data from sql / mt5 by switch the data source
        elif command == '-testCurrent':
            self.mt5Controller.pricesLoader.getPrices(
                symbols=['USDJPY'],
                count=1000,
                timeframe='15min'
            )

        elif command == '-testDeal':
            # define the dealer
            dealer = Dealer(self.mt5Controller, self.nodeJsApiController,
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
            historicalOrder = self.mt5Controller.get_historical_order(lastDays=3, position_id=338232986)
            historicalDeals = self.mt5Controller.get_historical_deals(lastDays=3, position_id=338232986)
            print()
        else:
            print_at('No command detected. Please input again. ')


"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
"""
