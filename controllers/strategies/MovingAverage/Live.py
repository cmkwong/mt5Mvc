from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
import time


class Live(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.onPosition = True
        self.LOT = 1

    def run(self, *,
            symbol: str = 'USDJPY',
            timeframe: str = '15min',
            fast_param: int = 21,
            slow_parm: int = 22,
            pt_sl: int = 100,
            pt_tp: int = 150,
            operation: str = 'long'
            ):
        # for calculating the sltp
        SLTP_FACTOR = 1 if operation == 'long' else -1
        while True:
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol], count=1000, timeframe=timeframe)
            MaData = self.getMaData(Prices, fast_param, slow_parm)
            # get signal by 'long' or 'short'
            signal = MaData[symbol][operation]
            if signal.iloc[-2] and not signal.iloc[-3]:
                # calculate the stop loss and take profit
                digit = Prices.all_symbols_info[symbol]['digits']
                sl = Prices.close[symbol][-1] + SLTP_FACTOR * (pt_sl * (10 ** (-digit)))
                tp = Prices.close[symbol][-1] - SLTP_FACTOR * (pt_tp * (10 ** (-digit)))
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
                self.mt5Controller.executor.request_execute(request)
                print(f'{symbol} open position.')

            # delay the operation
            time.sleep(5)
