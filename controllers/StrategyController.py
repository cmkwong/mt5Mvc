from myUtils.printModel import print_at
from models.Strategies.SwingScalping.Live_SwingScalping import Live_SwingScalping
import threading

class StrategyController:
    def __init__(self, mt5Controller, nodeJsServerController, defaultSymbols, tg=None):
        self.mt5Controller = mt5Controller
        self.nodeJsServerController = nodeJsServerController
        self.runningStrategies = []
        self.idleStrategies = {}    # ['name', 'strategy id']
        self.listLiveStrategies = [
            {'id': 0, 'name': Live_SwingScalping.__name__, 'class': Live_SwingScalping}
        ]
        self.Sybmols = defaultSymbols
        self.tg = tg

    # get list of strategies text
    def getListStrategiesText(self):
        txt = ''
        for id, strategy in enumerate(self.listLiveStrategies):
            txt += f"{id}. {strategy['name']}\n"
        return txt

    # run strategy
    def runThreadStrategy(self, strategyId, **kwargs):
        for strategy in self.listLiveStrategies:
            if strategy['id'] == strategyId:
                targetStrategy = strategy['class'](self.mt5Controller, self.nodeJsServerController, **kwargs)
                thread = threading.Thread(target=targetStrategy.run)
                thread.start()
                print_at(f"{strategy['name']} running ... with params: \n{kwargs}")
                self.runningStrategies.append(targetStrategy)

