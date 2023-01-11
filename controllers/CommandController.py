from myUtils.printModel import print_at
from myUtils import inputModel

from models import myParamModel

class CommandController:
    def __init__(self, mainController):
        self.mainController = mainController

    def run(self, command):
        # take the strategy into live and run
        if command == '-live':
            strategyListTxt = self.mainController.strategyController.getListStrategiesText(strategyOperation='live')
            strategyId = inputModel.askNum(f"{strategyListTxt}\nPlease input the index: ")
            strategyName = self.mainController.strategyController.strategiesList['live'][strategyId]['name']
            if inputModel.askConfirm("Do you want to run all of parameter?"): # ask for default all parameter or select specific parameter
                # loop for each parameter
                for params in myParamModel.STRATEGY_PARAMS['live'][strategyName]:
                    baseParam, runParam = params['base'], params['run']
                    # define the strategies
                    inventoryId = self.mainController.strategyController.appendStrategiesInventory(strategyId, 'live', **baseParam)
                    # run strategy on thread
                    self.mainController.strategyController.runThreadStrategy(inventoryId, **runParam)
            else:
                # ask which of parameter going to be selected
                paramTxt = myParamModel.getParamTxt(strategyName, 'live')
                paramId = inputModel.askNum(f"{paramTxt}\nPlease input the index: ")
                # get the selected param
                baseParam, runParam = myParamModel.getParamDic(strategyName, 'live', paramId)
                inventoryId = self.mainController.strategyController.appendStrategiesInventory(strategyId, 'live', **baseParam)
                self.mainController.strategyController.runThreadStrategy(inventoryId, **runParam)

        elif command == '-train':
            pass

        elif command == '-backtest':
            strategyTxt = self.mainController.strategyController.getListStrategiesText(strategyOperation='backtest')
            strategyId = inputModel.askNum(f"{strategyTxt}\nPlease input the index: ")
            inventoryId = self.mainController.strategyController.appendStrategiesInventory(strategyId, 'backtest', ask=True)
            self.mainController.strategyController.runThreadStrategy(inventoryId, ask=True)


        elif command == '-inventory':
            inventoryTxt = self.mainController.strategyController.getListStrategiesInventoryText()
            print_at(inventoryTxt, self.mainController.tg)

        # upload the data into mySql server
        elif command == '-upload':
            startTime = inputModel.askDate('2022-12-01 00:00', dateFormat='YYYY-MM-DD HH:mm')
            print_at(f"Start date set: {startTime}")
            endTime = inputModel.askDate('2022-12-31 23:59', dateFormat='YYYY-MM-DD HH:mm')
            print_at(f"End date set: {endTime}")
            self.mainController.nodeJsServerController.uploadDatas(self.mainController.defaultSymbols, startTime, endTime)
        else:
            print_at('No command detected. Please input again. ')
"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
3. 
"""