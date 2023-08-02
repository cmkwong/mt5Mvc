from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
"""
- Switch the dev and prod environment in MT5
-
"""


class Live(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.executor = mainController.mt5Controller.executor

    def run(self, *,
            symbols: SymbolList = 'USDJPY',
            timeframe: str = '15min',
            fast_param: int = 21,
            slow_parm: int = 22,
            operation: str = 'long'
            ):
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, count=1000, timeframe=timeframe)
        ma_data = self.get_ma_data(Prices, fast_param, slow_parm)
        print()
