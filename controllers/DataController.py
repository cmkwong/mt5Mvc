from controllers.myMT5.InitPrices import InitPrices
from models.myBacktest import exchgModel, pointsModel
from models.myUtils import dfModel
from models.myUtils.paramModel import SymbolList, DatetimeTuple

import collections
import abc

import MetaTrader5 as mt5
import pandas as pd


class DataController:
    def __init__(self, all_symbol_info: dict, deposit_currency: str='USD'):
        self.all_symbol_info = all_symbol_info
        self.deposit_currency = deposit_currency  # USD/GBP/EUR, main currency for deposit


    @abc.abstractmethod
    def _get_prices_df(self, *args, **kwargs):
        return NotImplemented





