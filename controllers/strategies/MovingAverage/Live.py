from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
import time


class Live(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.positionId = False
        self.lastPositionTime = False
        self.LOT = 1

    def run(self, *,
            symbol: str = 'USDJPY',
            timeframe: str = '15min',
            fast_param: int = 5,
            slow_parm: int = 22,
            pt_sl: int = 100,
            pt_tp: int = 210,
            operation: str = 'long'
            ):
        # for calculating the sltp
        SLTP_FACTOR = 1 if operation == 'long' else -1
        while True:
            # check if current position is closed
            if self.positionId and self.mt5Controller.orderFinished(self.positionId):
                print(f'{symbol} close position with position id: {self.positionId}')
                self.positionId = False

            # getting the Prices
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol], count=1000, timeframe=timeframe)
            MaData = self.getMaData(Prices, fast_param, slow_parm)
            MaData = self.getOperationGroup(MaData)
            # get the operation group value, either False / datetime
            operationGroupTime = MaData.loc[:, (symbol, f"{operation}_group")][-1]
            # get signal by 'long' or 'short'
            signal = MaData[symbol][operation]
            if not self.positionId:
                if signal.iloc[-2] and not signal.iloc[-3]:
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
                            lot=self.LOT
                        )
                        # execute request
                        Request = self.mt5Controller.executor.request_execute(request)
                        # if execute successful
                        if Request:
                            self.positionId = Request.order
                            self.lastPositionTime = signal.index[-1]
                            print(f'{symbol} open position with position id: {self.positionId}')
                        else:
                            print(f'{symbol} open position failed. ')
            # delay the operation
            time.sleep(5)
