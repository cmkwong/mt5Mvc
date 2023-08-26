from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
import time


class Live(Base):
    def __init__(self, mainController):
        self.nodeJsApiController = mainController.nodeJsApiController
        self.mt5Controller = mainController.mt5Controller
        self.openResult = None
        self.lastPositionTime = None
        self.LOT = 1

    @staticmethod
    def decodeParams(paramDf):

        # create the dict for positions
        param_positions = {}
        positionDf = paramDf.loc[:, ('paramid', 'pt', 'size')]
        for i, row in positionDf.iterrows():
            paramid = row['paramid']
            if paramid not in param_positions.keys():
                param_positions[paramid] = {float(row['pt']): float(row['size'])}
            else:
                param_positions[paramid][float(row['pt'])] = float(row['size'])
        # building the params
        params = {}
        for i, row in paramDf.iterrows():
            paramid = row['id']
            param = row.to_dict()
            if paramid in param_positions.keys():
                param['positions'] = param_positions[paramid]
            params[paramid] = param

        return params


    def run(self, *,
            symbol: str = 'USDJPY',
            timeframe: str = '15min',
            fast: int = 5,
            slow: int = 22,
            pt_sl: int = 100,
            pt_tp: int = 210,
            operation: str = 'long',
            lot: int = 1,
            positions: dict = None,
            **kwargs,
            ):
        # getting the partially close position
        # partiallyClosePositions = {}
        # for k, v in kwargs.items():
        #     if k[0:2] == 'pt' and v:
        #         partiallyClosePositions[float(k[2:]) / 100] = v
        # for calculating the sltp
        SLTP_FACTOR = 1 if operation == 'long' else -1
        while True:
            # check if current position is closed by sl or tp
            if self.openResult and self.mt5Controller.checkOrderClosed(self.openResult):
                # get the profit
                earn = self.mt5Controller.getPositionEarn(self.openResult)
                print(f'{symbol} position closed with position id: {self.openResult.order} and earn(sltp): {earn:2f}')
                self.openResult = None

            # getting the Prices and MaData
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol], count=1000, timeframe=timeframe)
            MaData = self.getMaData(Prices, fast, slow)
            MaData = self.getOperationGroup(MaData)

            # get the operation group value, either False / datetime
            operationGroupTime = MaData.loc[:, (symbol, f"{operation}_group")][-1]
            # get signal by 'long' or 'short' 1300
            signal = MaData[symbol][operation]

            if not self.openResult:
                if signal.iloc[-1] and not signal.iloc[-2]:
                    # to avoid open the position at same condition
                    if self.lastPositionTime != operationGroupTime:
                        # calculate the stop loss and take profit
                        digit = Prices.all_symbols_info[symbol]['digits']
                        sl = Prices.close[symbol][-1] - SLTP_FACTOR * (pt_sl * (10 ** (-digit)))
                        tp = Prices.close[symbol][-1] + SLTP_FACTOR * (pt_tp * (10 ** (-digit)))
                        # execute the open position
                        request = self.mt5Controller.executor.request_format(
                            symbol=symbol,
                            operation=operation,
                            sl=float(sl),
                            tp=float(tp),
                            deviation=5,
                            lot=lot
                        )
                        # execute request
                        self.openResult = self.mt5Controller.executor.request_execute(request)
                        # if execute successful
                        if self.mt5Controller.orderSentOk(self.openResult):
                            self.lastPositionTime = signal.index[-1]
                            print(f'{symbol} open position with position id: {self.openResult.order}')
                        else:
                            print(f'{symbol} open position failed. ')
            else:
                # check if signal should be close
                if not signal.iloc[-1] and signal.iloc[-2]:
                    request = self.mt5Controller.executor.close_request_format(self.openResult)
                    result = self.mt5Controller.executor.request_execute(request)
                    # get the profit
                    earn = self.mt5Controller.getPositionEarn(self.openResult)
                    # print(f'{symbol} close position with position id: {result.request.order}')
                    print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f}')
                    self.openResult = None

            # delay the operation
            time.sleep(5)
