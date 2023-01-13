from myUtils.printModel import print_at
from myUtils import paramModel

import config
import models.Strategies.SwingScalping as SwingScalping
from models import myParamModel
import threading


class StrategyController:
    def __init__(self, mt5Controller, nodeJsServerController, defaultSymbols, tg=None):
        self.mt5Controller = mt5Controller
        self.nodeJsServerController = nodeJsServerController
        self.strategiesInventory = {}  # [ id: { 'name': strategy name, 'instance': obj } ]
        self.strategiesList = {
            'live':
                {0:
                     {'name': SwingScalping.Live.__name__, 'class': SwingScalping.Live}
                 },
            'backtest':
                {0:
                     {'id': 0, 'name': SwingScalping.Backtest.__name__, 'class': SwingScalping.Backtest}
                 },
            'train':
                {0:
                     {'id': 0, 'name': SwingScalping.Train.__name__, 'class': SwingScalping.Train}
                 }
        }
        self.Sybmols = defaultSymbols
        self.tg = tg

    # get list of strategies text
    def getListStrategiesText(self, strategyOperation):
        txt = ''
        for id, strategy in self.strategiesList[strategyOperation].items():
            txt += f"{id}. {strategy['name']}\n"
        return txt

    # get list of defined strategies text
    def getListStrategiesInventoryText(self):
        """
        :return: text for strategiesInstance
        """
        txt = ''
        for id, strategyInventory in self.strategiesInventory.items():
            name = strategyInventory['instance'].getName
            txt += f"{id}. {name}({strategyInventory['type']}) Running: {strategyInventory['instance'].RUNNING}\n"
        return txt

    # define the strategies
    def appendStrategiesInventory(self, strategyId, strategyOperation, ask=False, **kwargs):
        inventoryId = len(self.strategiesInventory.keys())
        strategy = self.strategiesList[strategyOperation][strategyId]
        # if True then ask user to input the parameter
        if ask:
            kwargs = paramModel.ask_params(strategy['class'], config.PARAM_PATH, config.PARAM_FILE)
        instance = strategy['class'](self.mt5Controller, self.nodeJsServerController, **kwargs)
        self.strategiesInventory[inventoryId] = {
            'name': strategy['name'],
            'type': strategyOperation,
            'instance': instance,
        }
        print_at(f"{strategy['name']} with {kwargs} is being added. ")
        return inventoryId

    # run strategy from inventory with threading
    def runThreadStrategy(self, inventoryId, ask=False, **kwargs):
        """
        :param inventoryId: selected id from defined strategies
        :param kwargs: run() parameters
        """
        strategyInventory = self.strategiesInventory[inventoryId]
        if strategyInventory['instance'].RUNNING:
            print_at(f"{strategyInventory['name']} with cannot run again. ")
            return
        # if True then ask user to input the parameter
        if ask:
            kwargs = paramModel.ask_params(strategyInventory['instance'].run, config.PARAM_PATH, config.PARAM_FILE)
        # run the threading
        # thread = threading.Thread(target=strategyInventory['instance'].run, args=(*kwargs.values(), ))
        thread = threading.Thread(target=strategyInventory['instance'].run, kwargs=kwargs)
        thread.start()
        strategyInventory['instance'].RUNNING = True
        print_at(f"{strategyInventory['name']} running ... with params: \n{kwargs}")
