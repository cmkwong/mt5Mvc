from mt5Mvc.models.myUtils import paramModel, printModel, timeModel, inputModel
from mt5Mvc import config

import pandas as pd

class Handler_Analysis:
    def __init__(self, nodeJsApiController, mt5Controller, stockPriceLoader, threadController, strategyController, plotController, timeSeriesController, dfController):
        self.nodeJsApiController = nodeJsApiController
        self.mt5Controller = mt5Controller
        self.stockPriceLoader = stockPriceLoader
        self.threadController = threadController
        self.strategyController = strategyController
        self.plotController = plotController
        self.timeSeriesController = timeSeriesController
        self.dfController = dfController

    def run(self, command):
        # seeing the decomposition of time series
        if command == '-dct':
            symbol = 'USDJPY'
            paramFormat = {
                'symbols': [symbol],
                'start': (2023, 5, 30, 0, 0),
                'end': (2023, 9, 30, 23, 59),
                'timeframe': '15min'
            }
            param = paramModel.ask_param(**paramFormat)
            Prices = self.mt5Controller.pricesLoader.getPrices(**param)
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
            param = paramModel.ask_param(**paramFormat)
            Prices = self.mt5Controller.pricesLoader.getPrices(**param)
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
            ohlcvs_dfs = Prices.get_ohlcvs_from_prices()
            # get time string
            curTimeStr = timeModel.getTimeS(outputFormat="%Y%m%d-%H%M%S")
            for symbol, nextTargetDf in ohlcvs_dfs.items():
                X_gasf, X_gadf = self.timeSeriesController.getGAF(nextTargetDf['close'])
                self.plotController.getGafImg(X_gasf[0, :, :], X_gadf[0, :, :], nextTargetDf['close'], './docs/img', f"{curTimeStr}_{symbol}_gaf.jpg")

        # get the summary of df
        elif command == '-dfsumr':
            # read the df
            print("Read File csv/exsl: ")
            obj_readParam, readParam = paramModel.ask_param_fn(self.dfController.read_as_df)
            print("Output pdf: ")
            obj_sumParam, sumParam = paramModel.ask_param_fn(self.dfController.summaryPdf)
            _, nextTargetDf = obj_readParam(**readParam)
            obj_sumParam(nextTargetDf, **sumParam)

        else:
            return True