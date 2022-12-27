import config
from myUtils.printModel import print_at
from myUtils import inputModel

class CommandController:
    def __init__(self, mainController):
        self.mainController = mainController

    def run(self, command):
        if command == '-live':
            txt = self.mainController.strategyController.getListStrategiesText()
            print_at(f"{txt}\nPlease input the index: ", self.mainController.tg)
            usr_input = inputModel.num_input()
            for params in config.SWINGSCAPLING_PARAMS:
                self.mainController.strategyController.runThreadStrategy(usr_input, **params)


"""
1. List the strategy, ask for train, backtest, go-live
2. threading running command
3. 
"""