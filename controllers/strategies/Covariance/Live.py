from models.myUtils import printModel
from models.myUtils.paramModel import DatetimeTuple

import pandas as pd
import numpy as np

class Live:
    def __init__(self, mt5Controller, nodeJsServerController, tg=None):
        self.mt5Controller = mt5Controller
        self.nodeJsServerController = nodeJsServerController
        self.tg = tg

        # if running this strategy
        self.RUNNING = False

    def get_corelaDf(self, Prices, rowvar=False, bias=False):
        changes_arr = Prices.cc.values
        cor_matrix = np.corrcoef(changes_arr, rowvar=rowvar, bias=bias)
        symbols = list(Prices.cc.columns)
        corelaDf = pd.DataFrame(cor_matrix, index=symbols, columns=symbols)
        return corelaDf

    def run(self, *, symbol: str, start: DatetimeTuple, end: DatetimeTuple, timeframe: str):
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=self.mt5Controller.defaultSymbols, start=start, end=end, timeframe=timeframe)
        corelaDf = self.get_corelaDf(Prices)
        series = corelaDf[symbol].sort_values(ascending=False).drop(symbol)
        corelaDict = dict(series)
        print(f"{symbol} respect to: ")
        printModel.print_dict(corelaDict)
        return corelaDict




