from myUtils.printModel import print_at
from models.Strategies.SwingScalping.Live_SwingScalping import Live_SwingScalping
from models.Strategies.SwingScalping.Train_SwingScalping import Train_SwingScalping
from models.Strategies.SwingScalping.Backtest_SwingScalping import Backtest_SwingScalping
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
                     {'name': Live_SwingScalping.__name__, 'class': Live_SwingScalping}
                 },
            'backtest':
                {0:
                     {'id': 0, 'name': Backtest_SwingScalping.__name__, 'class': Backtest_SwingScalping}
                 },
            'train':
                {0:
                     {'id': 0, 'name': Train_SwingScalping.__name__, 'class': Train_SwingScalping}
                 }

        }
        self.Sybmols = defaultSymbols
        self.tg = tg

    # check what is the type of strategies
    # def _getListStrategies_DISCARD(self, strategyOperation):
    #     if strategyOperation == 'live':
    #         listStrategies = self.listLiveStrategies
    #     elif strategyOperation == 'train':
    #         listStrategies = self.listTrainStrategies
    #     elif strategyOperation == 'backtest':
    #         listStrategies = self.listBacktestStrategies
    #     else:
    #         print_at("Wrong strategy type. ", self.tg)
    #         raise Exception("Wrong strategy type")
    #     return listStrategies

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
            txt += f"{id}. {strategyInventory['name']}({strategyInventory['type']}) Running: {strategyInventory['instance'].RUNNING}\n"
        return txt

    # define the strategies
    def appendStrategiesInventory(self, strategyId, strategyOperation, **kwargs):
        inventoryId = len(self.strategiesInventory.keys())
        strategy = self.strategiesList[strategyOperation][strategyId]
        instance = strategy['class'](self.mt5Controller, self.nodeJsServerController, **kwargs)
        self.strategiesInventory[inventoryId] = {
            'name': strategy['name'],
            'type': strategyOperation,
            'instance': instance,
        }
        print_at(f"{strategy['name']} with {kwargs} is being added. ")
        return inventoryId

    # run strategy from inventory with threading
    def runThreadStrategy(self, inventoryId, **kwargs):
        """
        :param inventoryId: selected id from defined strategies
        :param kwargs: run() parameters
        """
        strategyInventory = self.strategiesInventory[inventoryId]
        if strategyInventory['instance'].RUNNING:
            print_at(f"{strategyInventory['name']} with {kwargs} cannot run again. ")
            return
        thread = threading.Thread(target=strategyInventory['instance'].run, args=(*kwargs.values(), ))
        thread.start()
        strategyInventory['instance'].RUNNING = True
        print_at(f"{strategyInventory['name']} running ... with params: \n{kwargs}")
