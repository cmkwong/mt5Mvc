from mt5Mvc.controllers.strategies.MovingAverage.Base import Base
from mt5Mvc.controllers.strategies.Dealer import Dealer
import time

class Live(Base, Dealer):
    def __init__(self, symbol: str = 'USDJPY', timeframe: str = '15min',
                 fast: int = 5, slow: int = 22,
                 pt_sl: int = 100, pt_tp: int = 210,
                 operation: str = 'long', lot: int = 1,
                 exitPoints: list = None,
                 strategy_name: str = '',
                 strategy_id: int = None,
                 **kwargs):
        super(Live, self).__init__(symbol=symbol, timeframe=timeframe,
                                   operation=operation, lot=lot, pt_sl=pt_sl, pt_tp=pt_tp, 
                                   exitPoints=exitPoints,
                                   strategy_name=strategy_name,
                                   strategy_id=strategy_id,
                                   strategy_detail=f'{fast}/{slow}'
                                   )
        # self.nodeJsApiController = nodeJsApiController
        # self.mt5Controller = mt5Controller
        self.lastPositionTime = None

        # run parameters
        self.fast = fast
        self.slow = slow

    # for printing the strategy
    def __str__(self):
        return f"{self.symbol}: {self.fast}/{self.slow}"

    @staticmethod
    def decodeParams(paramDf):
        """
        :return: dict: {strategy_id: { param } }
        """
        # create the dict for positions
        param_exits = {}
        positionDf = paramDf.loc[:, ('exit_id', 'strategy_id', 'pt', 'size')]
        for i, row in positionDf.iterrows():
            if not row['strategy_id']:
                continue
            strategy_id = row['strategy_id']
            if strategy_id not in param_exits.keys():
                # first time define the exit dict
                param_exits[strategy_id] = []
            param_exits[strategy_id].append([int(row['exit_id']), float(row['pt']), float(row['size'])])

        # building the params
        params = {}
        for i, row in paramDf.iterrows():
            strategy_id = row['strategy_id']
            # change dataframe row into dictionary
            param = row.to_dict()
            if strategy_id in param_exits.keys():
                param['exitPoints'] = param_exits[strategy_id]
            params[strategy_id] = param

        return params

    def run(self):

        while True:
            # check if current position is closed by sl or tp
            if self.position_id and self.mt5Controller.get_position_volume_balance(self.position_id) == 0:
                print("sub: ", self.position_id)
                self.checkDeal()
                # set to empty position
                self.position_id = None

            # getting the Prices and MaData
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[self.symbol], count=1000, timeframe=self.timeframe)
            MaData = self.getMaData(Prices, self.fast, self.slow)
            MaData = self.getOperationGroup_MaData(MaData)
            # getting curClose and its digit
            curClose = Prices.close[self.symbol][-1]
            self.digit = Prices.all_symbols_info[self.symbol]['digits']  # set the digit

            # get the operation group value, either False / datetime
            operationGroupTime = MaData.loc[:, (self.symbol, f"{self.operation}_group")][-1]
            # get signal by 'long' or 'short' 1300
            signal = MaData[self.symbol][self.operation]

            if not self.position_id:
                # open position signal
                if signal.iloc[-1] and not signal.iloc[-2]:
                    # to avoid open the position at same condition
                    if self.lastPositionTime != operationGroupTime:
                        # get the partially exit price
                        self.getExitPrices_tp(curClose)
                        # execute the open position
                        self.openDeal()
            else:
                # check if signal should be close
                if not signal.iloc[-1] and signal.iloc[-2]:
                    # close the deal
                    self.closeDeal(comment='Signal Off')

                # check each of partially position if being reached
                for exitPrice, data in self.exitPrices.items():
                    # setup data
                    exit_id = data['exit_id']
                    size = data['size']
                    point = data['point']
                    # bypass the exit position with size=0
                    if size == 0:
                        continue
                    # calculate the stop loss and take profit
                    if curClose >= exitPrice:
                        self.closeDeal_partial(size, info={'exit_id': exit_id})
                        # delete the position
                        self.exitPrices[exitPrice]['size'] = 0.0
                        # set to empty position if the balance is 0
                        if self.mt5Controller.get_position_volume_balance(self.position_id) == 0:
                            self.position_id = None
                        break

            # delay the operation
            time.sleep(5)
