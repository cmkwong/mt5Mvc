from models.myUtils.printModel import print_at
from models.myUtils import dicModel, paramModel

from controllers.strategies.SwingScalping.Live import Live as           SwingScalping_Live
from controllers.strategies.Covariance.Train import Train as            Covariance_Train
from controllers.strategies.Conintegration.Train import Train as        Cointegration_Train
from controllers.strategies.RL_Simple.Train import Train as             RL_Simple_Train

from controllers.DfController import DfController

import paramStorage

class CommandController:
    def __init__(self, mainController):
        self.mainController = mainController

    def run(self, command):
        # running SwingScalping_Live with all params
        if command == '-swL':
            defaultParams = paramStorage.METHOD_PARAMS['SwingScalping_Live']
            for defaultParam in defaultParams:
                strategy = SwingScalping_Live(self.mainController, auto=True)
                self.mainController.strategyController.runThreadFunction(strategy.run, **defaultParam)
                self.mainController.strategyController.appendRunning(dicModel.dic2Txt_k(defaultParam), strategy)

        # running Covariance_Live with all params
        elif command == '-cov':
            strategy = Covariance_Train(self.mainController)
            defaultParam = paramStorage.METHOD_PARAMS['Covariance_Live'][0]
            defaultParam = paramModel.ask_params(strategy.run, defaultParam)
            self.mainController.strategyController.runThreadFunction(strategy.run, **defaultParam)

        elif command == '-coinT':
            strategy = Cointegration_Train(self.mainController)
            defaultParam = paramStorage.METHOD_PARAMS['Cointegration_Train'][0]
            defaultParam = paramModel.ask_params(strategy.simpleCheck, defaultParam)
            self.mainController.strategyController.runThreadFunction(strategy.simpleCheck, **defaultParam)

        elif command == '-rlT':
            strategy = RL_Simple_Train(self.mainController)
            strategy.run()

        # upload the data into mySql server
        elif command == '-upload':
            # setup the source into mt5
            originalSource = self.mainController.mt5Controller.mt5PricesLoader.source
            self.mainController.mt5Controller.mt5PricesLoader.source = 'mt5'
            # upload Prices
            param = paramStorage.METHOD_PARAMS['upload_mt5_getPrices'][0]
            param = paramModel.ask_params(self.mainController.mt5Controller.mt5PricesLoader.getPrices, param)
            Prices = self.mainController.mt5Controller.mt5PricesLoader.getPrices(**param)
            self.mainController.nodeJsApiController.uploadOneMinuteForexData(Prices)
            # resume to original source
            self.mainController.mt5Controller.mt5PricesLoader.source = originalSource

        # all symbol info upload
        elif command == '-symbol':
            # upload all_symbol_info
            all_symbol_info = self.mainController.mt5Controller.mt5PricesLoader.all_symbol_info
            param = paramStorage.METHOD_PARAMS['upload_all_symbol_info'][0]
            param = paramModel.ask_params(self.mainController.nodeJsController.apiController.uploadAllSymbolInfo, param)
            param['all_symbol_info'] = all_symbol_info
            self.mainController.nodeJsController.apiController.uploadAllSymbolInfo(**param)

        # switch the nodeJS server env: dev / prod
        elif command == '-env':
            self.mainController.nodeJsApiController.switchEnv()

        # switch the price loader source from mt5 / local
        elif command == '-source':
            self.mainController.mt5Controller.mt5PricesLoader.switchSource()

        # get the summary of df
        elif command == '-dfsumr':
            dfController = DfController()
            # read the df
            readParam = paramModel.ask_params(DfController.readAsDf, {'filename': 'Health_conditions_among_children_under_age_18__by_selected_characteristics__United_States.csv'})
            sumParam = paramModel.ask_params(DfController.summaryPdf, {'filename': 'Health_conditions_among_children_under_age_18__by_selected_characteristics__United_States.pdf'})
            df = dfController.readAsDf(**readParam)
            dfController.summaryPdf(df, **sumParam)
        else:
            print_at('No command detected. Please input again. ')
"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
"""