import config
from myUtils.printModel import print_at
from myUtils import inputModel

from models import myParamModel

class CommandController:
    def __init__(self, mainController):
        self.mainController = mainController

    def run(self, command):
        # take the strategy into live and run
        if command == '-live':
            txt = self.mainController.strategyController.getListStrategiesText(strategyOperation='live')
            strategyId = inputModel.askNum(f"{txt}\nPlease input the index: ")
            strategyName = self.mainController.strategyController.strategiesList['live'][strategyId]['name']
            # loop for each parameter
            for params in myParamModel.STRATEGY_PARAMS['live'][strategyName]:
                baseParam, runParam = params['base'], params['run']
                # define the strategies
                inventoryId = self.mainController.strategyController.appendStrategiesInventory(strategyId, 'live', **baseParam)
                # run strategy on thread
                self.mainController.strategyController.runThreadStrategy(inventoryId, **runParam)

        elif command == '-train':
            pass

        elif command == '-backtest':
            txt = self.mainController.strategyController.getListStrategiesText(strategyOperation='backtest')
            usr_input = inputModel.askNum(f"{txt}\nPlease input the index: ")
            startTime = inputModel.askDate('2022-12-01 00:00', dateFormt='YYYY-MM-DD HH:mm')
            print_at(f"Start date set: {startTime}")
            endTime = inputModel.askDate('2022-12-22 23:59', dateFormt='YYYY-MM-DD HH:mm')
            print_at(f"End date set: {endTime}")
            lot = inputModel.askNum("Please input the lot")

        # upload the data into mySql server
        elif command == '-upload':
            startTime = inputModel.askDate('2022-12-01 00:00', dateFormt='YYYY-MM-DD HH:mm')
            print_at(f"Start date set: {startTime}")
            endTime = inputModel.askDate('2022-12-22 23:59', dateFormt='YYYY-MM-DD HH:mm')
            print_at(f"End date set: {endTime}")
            self.mainController.nodeJsServerController.uploadDatas(self.mainController.defaultSymbols, startTime, endTime)
        else:
            print_at('No command detected. Please input again. ')
"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
3. 
"""