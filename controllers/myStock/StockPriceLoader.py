from controllers.myMT5.InitPrices import InitPrices
from models.myBacktest import exchgModel
from models.myUtils import timeModel, inputModel
from models.myUtils.paramModel import SymbolList, DatetimeTuple
import config

from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

class StockPriceLoader:
    def __init__(self, nodeJsApiController):
        self.nodeJsApiController = nodeJsApiController

    def getPrices(self, *,
                  symbols: str = 'AMZ',
                  start: DatetimeTuple = (2023, 6, 1, 0, 0),
                  end: DatetimeTuple = (2023, 6, 30, 23, 59),
                  timeframe: str = '15min',
                  count: int = 0,
                  ohlcvs: str = '111111'):

        pass