from models.myUtils.printModel import print_at
from models.myUtils import inputModel, dicModel
from models import myParams

from controllers.strategies.SwingScalping.Live import Live as           SwingScalping_Live
from controllers.strategies.SwingScalping.Backtest import Backtest as   SwingScalping_Backtest
from controllers.strategies.SwingScalping.Train import Train as         SwingScalping_Train
from controllers.strategies.Covariance.Live import Live as              Covariance_Live
from controllers.strategies.RL_Simple.Train import Train as             RL_Train

class CommandController:
    def __init__(self, mainController):
        self.mainController = mainController

    def run(self, command):
        # running SwingScalping_Live with all params
        if command == '-livesw':
            for param in myParams.METHOD_PARAMS['SwingScalping_Live']:
                object = SwingScalping_Live(self.mainController, auto=True)
                self.mainController.strategyController.runThreadFunction(object.run, **param)
                self.mainController.strategyController.appendRunning(dicModel.dic2Txt_k(param), object)

        # running Covariance_Live with all params
        elif command == '-cov':
            object = Covariance_Live(self.mainController)
            param = myParams.METHOD_PARAMS['Covariance_Live'][0]
            self.mainController.strategyController.runThreadFunction(object.run, **param)

        # take the strategy into live and run
        elif command == '-_live':
            strategyListTxt = self.mainController.strategyController.getListStrategiesText(strategyOperation='live')
            strategyId = inputModel.askNum(f"{strategyListTxt}\nPlease input the index: ")
            strategyName = self.mainController.strategyController.strategiesList['live'][strategyId]['name']
            if inputModel.askConfirm("Do you want to run all of parameter?"): # ask for default all parameter or select specific parameter
                self.mainController.strategyController.runAllParam(strategyId, strategyName, 'live')
            else:
                self.mainController.strategyController.selectOneParam(strategyId, strategyName, 'live')

        elif command == '-train':
            strategyListTxt = self.mainController.strategyController.getListStrategiesText(strategyOperation='train')
            strategyId = inputModel.askNum(f"{strategyListTxt}\nPlease input the index: ")
            strategyName = self.mainController.strategyController.strategiesList['train'][strategyId]['name']
            if inputModel.askConfirm("Do you want to run all of parameter?"):  # ask for default all parameter or select specific parameter
                self.mainController.strategyController.runAllParam(strategyId, strategyName, 'train')
            else:
                self.mainController.strategyController.selectOneParam(strategyId, strategyName, 'train')

        elif command == '-backtest':
            strategyTxt = self.mainController.strategyController.getListStrategiesText(strategyOperation='backtest')
            strategyId = inputModel.askNum(f"{strategyTxt}\nPlease input the index: ")
            inventoryId = self.mainController.strategyController.appendStrategiesInventory(strategyId, 'backtest', ask=True)
            self.mainController.strategyController.runThreadStrategy(inventoryId, ask=False)


        elif command == '-inventory':
            inventoryTxt = self.mainController.strategyController.getListStrategiesInventoryText()
            print_at(inventoryTxt, self.mainController.tg)

        # upload the data into mySql server
        elif command == '-upload':
            startTime = inputModel.askDate(defaultDate='2023-02-01 00:00', dateFormat='%Y-%m-%d %H:%M')
            print_at(f"Start date set: {startTime}")
            endTime = inputModel.askDate(defaultDate='2023-02-25 23:59', dateFormat='%Y-%m-%d %H:%M')
            print_at(f"End date set: {endTime}")
            self.mainController.nodeJsServerController.uploadDatas(self.mainController.defaultSymbols, startTime, endTime)
        else:
            print_at('No command detected. Please input again. ')
"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
3. 
"""