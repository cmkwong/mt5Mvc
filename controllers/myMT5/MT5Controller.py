from controllers.myMT5.MT5PricesLoader import MT5PricesLoader
from controllers.myMT5.MT5Executor import MT5Executor
from controllers.myMT5.BaseMt5 import BaseMt5

class MT5Controller(BaseMt5):
    def __init__(self, timezone='Hongkong', deposit_currency='USD', type_filling='ioc'):
        super().__init__()
        self.executor = MT5Executor(type_filling)  # execute the request (buy/sell)
        self.pricesLoader = MT5PricesLoader(self.all_symbol_info, timezone, deposit_currency)  # loading the loader
