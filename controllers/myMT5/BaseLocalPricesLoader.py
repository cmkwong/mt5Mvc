import os
import pandas as pd
import numpy as np

from models.myUtils import fileModel
from controllers.myMT5.BasePricesLoader import BasePricesLoader

class BaseLocalPricesLoader(BasePricesLoader):

    def __init__(self, main_path):
        """
        :param main_path: str
        """
        self.main_path = main_path

    def _get_names_and_usecols(self, ohlc):
        """
        note 84e
        :param ohlc: str, eg: '1001'
        :return:    names, [str], names assigned to columns
                    usecols, int that column will be used
        """
        type_names = ['open', 'high', 'low', 'close']
        names = ['time']
        usecols = [0]
        for i, code in enumerate(ohlc):
            if code == '1':
                names.append(type_names[i])
                usecols.append(i + 1)
        return names, usecols

    def read_MyCSV(self, file_name, data_time_difference_to_UTC, names, usecols, brokerTimeBetweenUtc=2):
        """
        the timezone is Eastern Standard Time (EST) time-zone WITHOUT Day Light Savings adjustments
        :param file_name: str
        :param time_difference_in_hr: time difference between current broker
        :param ohlcvs: str, eg: '1001'
        :return: pd.DataFrame
        """
        shifted_hr = brokerTimeBetweenUtc + data_time_difference_to_UTC
        full_path = os.path.join(self.main_path, file_name)
        df = pd.read_csv(full_path, header=None, names=names, usecols=usecols)
        df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index).shift(shifted_hr, freq='H')
        return df

    def read_symbol_price(self, symbol, data_time_difference_to_UTC, ohlcvs='1001'):
        """
        :param main_path: str, file path that contains several minute excel loader
        :param data_time_difference_to_UTC: int, the time difference between downloaded loader and broker
        :param timeframe: str, '1H'
        :param ohlcvs: str, '1001'
        :return: pd.DataFrame, symbol_prices
        """
        symbol_prices = None
        names, usecols = self._get_names_and_usecols(ohlcvs)
        symbol_path = os.path.join(self.main_path, symbol)
        min_data_names = fileModel.getFileList(symbol_path)
        # concat a symbol in a dataframe (axis = 0)
        for file_count, file_name in enumerate(min_data_names):
            df = self.read_MyCSV(file_name, data_time_difference_to_UTC, names, usecols)
            if file_count == 0:
                symbol_prices = df.copy()
            else:
                symbol_prices = pd.concat([symbol_prices, df], axis=0)
        # drop the duplicated index row
        symbol_prices = symbol_prices[~symbol_prices.index.duplicated(keep='first')]  # note 80b and note 81c
        return symbol_prices

    def write_min_extra_info(self, file_name, symbols, long_signal, short_signal, long_modify_exchg_q2d, short_modify_exchg_q2d):
        """
        :param file_name: str
        :param symbols: list
        :param long_signal: pd.Series
        :param short_signal: pd.Series
        :param long_modify_exchg_q2d: pd.DataFrame
        :param short_modify_exchg_q2d: pd.DataFrame
        :return: None
        """
        # concat the loader axis=1
        df_for_min = pd.concat([long_signal, short_signal, long_modify_exchg_q2d, short_modify_exchg_q2d], axis=1)
        # re-name
        level_2_arr = np.array(['long', 'short'] + symbols * 2)
        level_1_arr = np.array(['signal'] * 2 + ['long_q2d'] * len(symbols) + ['short_q2d'] * len(symbols))
        df_for_min.columns = [level_1_arr, level_2_arr]
        df_for_min.to_csv(os.path.join(self.main_path, file_name))
        print("Extra info write to {}".format(self.main_path))

    def read_min_extra_info(self):
        """
        :param col_list: list, [str/int]: required column names
        :return: Series, Series, DataFrame, DataFrame
        """
        file_names = fileModel.getFileList(self.main_path, reverse=True)
        dfs = None
        for i, file_name in enumerate(file_names):
            full_path = os.path.join(self.main_path, file_name)
            df = pd.read_csv(full_path, header=[0, 1], index_col=0)
            if i == 0:
                dfs = df.copy()
            else:
                dfs = pd.concat([dfs, df], axis=0)
        # di-assemble into different parts
        long_signal = dfs.loc[:, ('signal', 'long')]
        short_signal = dfs.loc[:, ('signal', 'short')]
        long_q2d = dfs.loc[:, ('long_q2d')]
        short_q2d = dfs.loc[:, ('short_q2d')]
        return long_signal, short_signal, long_q2d, short_q2d

    def _get_local_prices(self, symbols, data_time_difference_to_UTC, ohlcvs):
        """
        :param symbols: [str]
        :param data_time_difference_to_UTC: int
        :param timeframe: str, eg: '1H', '1min'
        :param ohlcvs: str, eg: '1001'
        :return: pd.DataFrame
        """
        prices_df = pd.DataFrame()
        for i, symbol in enumerate(symbols):
            print("Processing: {}".format(symbol))
            price_df = self.read_symbol_price(symbol, data_time_difference_to_UTC, ohlcvs=ohlcvs)
            if i == 0:
                prices_df = price_df.copy()
            else:
                # join='outer' method with all symbols in a bigger dataframe (axis = 1)
                prices_df = pd.concat([prices_df, price_df], axis=1, join='outer')  # because of 1 minute loader and for ensure the completion of loader, concat in join='outer' method

        # replace NaN values with preceding values
        prices_df.fillna(method='ffill', inplace=True)
        prices_df.dropna(inplace=True, axis=0)

        # get prices in dict
        prices = self._prices_df2dict(prices_df, symbols, ohlcvs)

        return prices