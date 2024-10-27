from mt5Mvc.controllers.BasePriceLoader import BasePriceLoader
from mt5Mvc.controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from mt5Mvc.controllers.DfController import DfController

import config

import pandas as pd

class StockPriceLoader(BasePriceLoader):
    def __init__(self):
        super().__init__()
        self.nodeJsApiController = NodeJsApiController()
        # loading the local data, if selected
        self.local_path = None

    def _get_prices_df(self,
                       symbols: list = config.Default_Stock_Symbols,
                       timeframe: str = '15min',
                       start: tuple = (2023, 6, 1, 0, 0),
                       end: tuple = (2023, 6, 30, 23, 59),
                       ohlcvs: str = '111111'):

        join = 'outer'
        prices_df = None
        required_types = self._price_type_from_code(ohlcvs)
        for i, symbol in enumerate(symbols):
            if self.data_source == 'local':
                price = None
            else:
                price = self.nodeJsApiController.downloadSeriesData('stock', symbol, timeframe, start, end)
            price = price.loc[:, required_types]
            if i == 0:
                prices_df = price.copy()
            else:
                prices_df = pd.concat([prices_df, price], axis=1, join=join)

        # replace NaN values with preceding values
        prices_df.fillna(method='ffill', inplace=True)
        prices_df.dropna(how='all', inplace=True, axis=0)

        # get prices in dict
        prices = self._prices_df2dict(prices_df, symbols, ohlcvs)

        return prices

    def getPrices(self, *,
                  symbols: list = config.Default_Stock_Symbols,
                  start: tuple = (2023, 1, 1, 0, 0),
                  end: tuple = (2023, 6, 30, 23, 59),
                  timeframe: str = '15min',
                  ohlcvs: str = '111111'
                  ):
        prices = self._get_prices_df(symbols, timeframe, start, end, ohlcvs)
        Prices = self.get_Prices_format(symbols, prices, ohlcvs)

        return Prices
