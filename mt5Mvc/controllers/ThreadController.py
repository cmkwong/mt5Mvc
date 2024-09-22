from mt5Mvc.models.myUtils.printModel import print_at

import threading

class ThreadController:
    # def __init__(self):
    #     self.RUNNINGS = {}
    #
    # def appendRunning(self, name, object):
    #     self.RUNNINGS[name] = object

    def runThreadFunction(self, fn, **kwargs):
        """
        :param inventoryId: selected id from defined strategies
        :param kwargs: run() parameters
        """
        # run the threading
        # thread = threading.Thread(target=strategyInventory['instance'].run, args=(*kwargs.values(), ))
        thread = threading.Thread(target=fn, kwargs=kwargs)
        thread.start()
        if not kwargs:
            print_at(f"{fn} running ...")
        else:
            print_at(f"{fn} running ... with params: \n{kwargs}")

    def getThreadingFunction(self):
        for thread in threading.enumerate():
            print(thread.name)
# class StrategyController_DISCARD:
#     def __init__(self, mt5Controller, nodeJsServerController, defaultSymbols, tg=None):
#         self.mt5Controller = mt5Controller
#         self.nodeJsServerController = nodeJsServerController
#         self.strategiesInventory = {}  # [ id: { 'name': strategy name, 'instance': obj } ]
#         self.strategiesList = {
#             'live':
#                 {
#                     0: {'name': fileModel.getParentFolderName(SwingScalping_Live),          'class': SwingScalping_Live},
#                     1: {'name': fileModel.getParentFolderName(Covariance_Live),             'class': Covariance_Live},
#                 },
#             'backtest':
#                 {
#                     0: {'name': fileModel.getParentFolderName(SwingScalping_Backtest),      'class': SwingScalping_Backtest}
#                 },
#             'train':
#                 {
#                     0: {'name': fileModel.getParentFolderName(SwingScalping_Train),         'class': SwingScalping_Train},
#                     1: {'name': fileModel.getParentFolderName(RL_Train),                    'class': RL_Train}
#                 }
#         }
#         self.Sybmols = defaultSymbols
#         self.tg = tg
#
#     # run all the parameter
#     def runAllParam(self, strategyId, strategyName, strategyOperation):
#         # loop for each parameter
#         for params in myParams.STRATEGY_PARAMS_DISCARD[strategyOperation][strategyName]:
#             baseParam, runParam = params['base'], params['run']
#             # define the strategies
#             inventoryId = self.appendStrategiesInventory(strategyId, strategyOperation, **baseParam)
#             # run strategy on thread
#             self.runThreadStrategy(inventoryId, **runParam)
#
#     def selectOneParam(self, strategyId, strategyName, strategyOperation):
#         # ask which of parameter going to be selected
#         paramTxt = myParams.getParamTxt(strategyName, strategyOperation)
#         paramId = inputModel.askNum(f"{paramTxt}\nPlease input the index: ")
#         # get the selected param
#         baseParam, runParam = myParams.getParamDic(strategyName, strategyOperation, paramId)
#         inventoryId = self.appendStrategiesInventory(strategyId, strategyOperation, **baseParam)
#         self.runThreadStrategy(inventoryId, **runParam)
#
#     # get list of strategies text
#     def getListStrategiesText(self, strategyOperation):
#         txt = ''
#         for id, strategy in self.strategiesList[strategyOperation].items():
#             txt += f"{id}. {strategy['name']}\n"
#         return txt
#
#     # get list of defined strategies text
#     def getListStrategiesInventoryText(self):
#         """
#         :return: text for strategiesInstance
#         """
#         txt = ''
#         for id, strategyInventory in self.strategiesInventory.items():
#             identity = strategyInventory['instance'].getIdentity
#             txt += f"{id}. {identity}({strategyInventory['type']}) Running: {strategyInventory['instance'].RUNNING}\n"
#         return txt
#
#     # define the strategies
#     def appendStrategiesInventory(self, strategyId, strategyOperation, ask=False, **kwargs):
#         inventoryId = len(self.strategiesInventory.keys())
#         strategy = self.strategiesList[strategyOperation][strategyId]
#         # if True then ask user to input the parameter
#         if ask:
#             kwargs = paramModel.ask_txtParams(strategy['class'], config.PARAM_PATH, config.PARAM_FILE)
#         instance = strategy['class'](self.mt5Controller, self.nodeJsServerController, **kwargs)
#         self.strategiesInventory[inventoryId] = {
#             'name': strategy['name'],
#             'type': strategyOperation,
#             'instance': instance,
#         }
#         print_at(f"{strategy['name']} with {kwargs} is being added. ")
#         return inventoryId
#
#     # run strategy from inventory with threading
#     def runThreadStrategy(self, inventoryId, ask=False, **kwargs):
#         """
#         :param inventoryId: selected id from defined strategies
#         :param kwargs: run() parameters
#         """
#         strategyInventory = self.strategiesInventory[inventoryId]
#         if strategyInventory['instance'].RUNNING:
#             print_at(f"{strategyInventory['name']} with cannot run again. ")
#             return
#         # if True then ask user to input the parameter
#         if ask:
#             kwargs = paramModel.ask_txtParams(strategyInventory['instance'].run, config.PARAM_PATH, config.PARAM_FILE)
#         # run the threading
#         # thread = threading.Thread(target=strategyInventory['instance'].run, args=(*kwargs.values(), ))
#         thread = threading.Thread(target=strategyInventory['instance'].run, kwargs=kwargs)
#         thread.start()
#         strategyInventory['instance'].RUNNING = True
#         print_at(f"{strategyInventory['name']} running ... with params: \n{kwargs}")
