from models.myUtils.paramModel import SymbolList, DatetimeTuple
from models.myUtils import printModel
from controllers.strategies.MovingAverage.Base import Base
from controllers.strategies.Dealer import Dealer
import time


class Live(Base, Dealer):
    def __init__(self, mainController, *,
                 symbol: str = 'USDJPY', timeframe: str = '15min',
                 fast: int = 5, slow: int = 22,
                 pt_sl: int = 100, pt_tp: int = 210,
                 operation: str = 'long', lot: int = 1,
                 exitPoints: dict = None,
                 strategy_name: str = '',
                 strategy_id: int = None,
                 **kwargs):
        super(Live, self).__init__(mainController, 
                                   symbol=symbol, timeframe=timeframe, 
                                   operation=operation, lot=lot, pt_sl=pt_sl, pt_tp=pt_tp, 
                                   exitPoints=exitPoints,
                                   strategy_name=strategy_name, strategy_id=strategy_id, strategy_detail=f'{fast}/{slow}'
                                   )
        self.nodeJsApiController = mainController.nodeJsApiController
        self.mt5Controller = mainController.mt5Controller
        self.openResult = None
        self.lastPositionTime = None
        self.exitPrices = None  # for partially close the position (take profit)
        self.digit = None

        # run parameters
        self.fast = fast
        self.slow = slow

    @staticmethod
    def decodeParams(paramDf):

        # create the dict for positions
        param_exits = {}
        positionDf = paramDf.loc[:, ('strategy_id', 'pt', 'size')]
        for i, row in positionDf.iterrows():
            if not row['strategy_id']:
                continue
            strategy_id = row['strategy_id']
            if strategy_id not in param_exits.keys():
                # first time define the exit dict
                param_exits[strategy_id] = {float(row['pt']): float(row['size'])}
            else:
                # assign another exit points
                param_exits[strategy_id][float(row['pt'])] = float(row['size'])
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
            if self.openResult and self.mt5Controller.check_order_closed(self.openResult.order) == 0:
                self.checkDeal()
                # set to empty position
                self.openResult = None

            # getting the Prices and MaData
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[self.symbol], count=1000, timeframe=self.timeframe)
            MaData = self.getMaData(Prices, self.fast, self.slow)
            MaData = self.getOperationGroup(MaData)
            # getting curClose and its digit
            curClose = Prices.close[self.symbol][-1]
            self.digit = Prices.all_symbols_info[self.symbol]['digits']  # set the digit

            # get the operation group value, either False / datetime
            operationGroupTime = MaData.loc[:, (self.symbol, f"{self.operation}_group")][-1]
            # get signal by 'long' or 'short' 1300
            signal = MaData[self.symbol][self.operation]

            if not self.openResult:
                # open position signal
                if signal.iloc[-1] and not signal.iloc[-2]:
                    # to avoid open the position at same condition
                    if self.lastPositionTime != operationGroupTime:
                        # get the partially exit price
                        self.exitPrices = self.getExitPrices_tp(curClose)
                        # execute the open position
                        self.openDeal()
            else:
                # check if signal should be close
                if not signal.iloc[-1] and signal.iloc[-2]:
                    # close the deal
                    self.closeDeal()

                # check if the partially position being reached
                for exitPrice, data in self.exitPrices.items():
                    size = data['size']
                    point = data['point']
                    # check if available size left
                    if size > 0:
                        # calculate the stop loss and take profit
                        if curClose >= exitPrice:
                            # # print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f} (Partial) and time taken {duration}')
                            self.closeDeal_partial(size)
                            # reset the position
                            self.exitPrices[exitPrice] = 0

            # delay the operation
            time.sleep(5)
