from models.myUtils.paramModel import DatetimeTuple
import config

import pandas as pd
import numpy as np

class Train:
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsServerController
        self.symbols = config.DefaultSymbols

    def get_corelaDf(self, Prices, rowvar=False, bias=False):
        changes_arr = Prices.cc.values
        cor_matrix = np.corrcoef(changes_arr, rowvar=rowvar, bias=bias)
        symbols = list(Prices.cc.columns)
        corelaDf = pd.DataFrame(cor_matrix, index=symbols, columns=symbols)
        return corelaDf

    def run(self, *, start: DatetimeTuple, end: DatetimeTuple, timeframe: str):
        Prices = self.mt5Controller.mt5PricesLoader.getPrices(symbols=self.symbols, start=start, end=end, timeframe=timeframe)
        corelaDf = self.get_corelaDf(Prices)
        corelaTxtDf = pd.DataFrame()
        for symbol in self.symbols:
            series = corelaDf[symbol].sort_values(ascending=False).drop(symbol)
            corelaDict = dict(series)
            corelaTxtDf[symbol] = pd.Series([f"{key}: {value * 100:.2f}%" for key, value in corelaDict.items()])
            # printModel.print_dict(corelaDict)
        pd.set_option('display.max_columns', None)
        print("\n\n")
        print(corelaTxtDf)
        pd.reset_option("display.max_columns")

