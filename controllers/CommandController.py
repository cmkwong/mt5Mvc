import config
from myUtils.printModel import print_at
from myUtils import inputModel

class CommandController:
    def __init__(self, mainController):
        self.mainController = mainController

    def run(self, command):
        # take the strategy into live
        if command == '-live':
            txt = self.mainController.strategyController.getListStrategiesText()
            print_at(f"{txt}\nPlease input the index: ", self.mainController.tg)
            usr_input = inputModel.askNum()
            for params in config.SWINGSCAPLING_PARAMS:
                self.mainController.strategyController.runThreadStrategy(usr_input, **params)

        elif command == '-train':
            pass

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