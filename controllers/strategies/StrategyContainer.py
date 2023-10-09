
class StrategyContainer:

    def __init__(self, mt5Controller, nodeJsApiController):
        self.mt5Controller = mt5Controller
        self.nodeJsApiController = nodeJsApiController
        # storing the running strategy
        self.RunningStrategies = {} # id: strategy instance

    def add(self, obj):
        id = obj.strategy_id
        self.RunningStrategies[id] = obj

    # if exist strategy, return True
    def exist(self, strategy_id):
        if strategy_id in self.RunningStrategies.keys():
            return True
        return False

    def load_param(self, strategy_name):
        """
        load the parameter from Database by position_id with respect to the strategy type
        :param strategy_name: str
        :return:
        """
        positionsDf = self.mt5Controller.get_active_positions()
        for i, row in positionsDf.iterrows():
            position_id = row['ticket']
            price_open = row['price_open']
            url = self.nodeJsApiController.strategyParamUrl
            param = {
                'strategy_name': strategy_name,
                'position_id': position_id
            }
            paramDf = self.nodeJsApiController.getDataframe(url, param)
            if paramDf.empty:
                print(f"{position_id} has no param found. ")
                continue
            yield paramDf, position_id, price_open

    # iter the strategies
    def __iter__(self):
        for id, strategy in self.RunningStrategies.items():
            yield id, strategy
