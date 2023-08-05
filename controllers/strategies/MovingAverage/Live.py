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
            operation: str = 'long'
            ):
        while True:
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol], count=1000, timeframe=timeframe)
            ma_data = self.getMaData(Prices, fast_param, slow_parm)
            signal = ma_data[symbol][operation]
            if signal.iloc[-2] and not signal.iloc[-3]:
                # execute the open position
                request = self.mt5Controller.executor.request_format(
                    symbol=symbol,
                    operation=operation,
                    sl=float(self.status['sl']),
                    tp=float(self.status['tp']),
                    deviation=5,
                    lot=self.LOT
                )
                # execute request
                self.mt5Controller.executor.request_execute(request)
                print(f'{symbol} open position.')
            elif not signal.iloc[-2] and signal.iloc[-3]:
                # execute the close position
                print(f'{symbol} close position.')

            # delay the operation
            time.sleep(5)
