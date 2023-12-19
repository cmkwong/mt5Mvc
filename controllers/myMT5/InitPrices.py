from models.myUtils import dfModel

from dataclasses import dataclass
import pandas as pd

@dataclass
class InitPrices:
    def __init__(self,
                symbols: list,
                close: pd.DataFrame,
                cc: pd.DataFrame,
                all_symbols_info: dict = None,
                quote_exchg: pd.DataFrame = pd.DataFrame(),
                base_exchg: pd.DataFrame = pd.DataFrame(),
                open: pd.DataFrame = pd.DataFrame(),
                high: pd.DataFrame = pd.DataFrame(),
                low: pd.DataFrame = pd.DataFrame(),
                volume: pd.DataFrame = pd.DataFrame(),
                spread: pd.DataFrame = pd.DataFrame()
                ):
        self.symbols = symbols
        self.all_symbols_info = all_symbols_info if all_symbols_info else {}
        self.close = close
        self.cc = cc
        self.quote_exchg = quote_exchg
        self.base_exchg = base_exchg
        self.open = open
        self.high = high
        self.low = low
        self.volume = volume
        self.spread = spread
        self.ptD = self.get_points_dff_df(ptValue_need=False, depositValue_need=False) if all_symbols_info else None
        self.ptDv = self.get_points_dff_df() if all_symbols_info else None # in-deposit, eg USD
        # get attr
        self.attrs = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    def getValidCols(self):
        validCol = []
        for name in self.attrs:
            value = getattr(self, name)
            if not value.empty:
                validCol.append(value)
        return validCol

    def split_Prices(self, percentage):
        trainDict, testDict = {}, {}
        for name in self.attrs:
            attr = getattr(self, name)
            if not isinstance(attr, pd.DataFrame):
                trainDict[name] = attr
                testDict[name] = attr
            else:
                trainDict[name], testDict[name] = dfModel.split_df(attr, percentage)
        return InitPrices(**trainDict), InitPrices(**testDict)

    # get {'symbol': df with ohlcvs}
    def getOhlcvsFromPrices(self):
        """
        resume into normal dataframe
        :param symbols: [symbol str]
        :param Prices: Prices collection
        :return: {pd.DataFrame}
        """
        ohlcsvs = {}
        nameDict = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume', 'spread': 'spread', 'quote_exchg': 'quote_exchg', 'base_exchg': 'base_exchg'}
        for si, symbol in enumerate(self.symbols):
            requiredDf = pd.DataFrame()  # create empty df
            for name in self.attrs:  # name = variable name; field = pd.dataframe/ value
                # only need the cols in nameDict
                if name not in nameDict.keys(): continue
                # get the attr
                attr = getattr(self, name)
                # if not dataframe
                if not isinstance(attr, pd.DataFrame): continue
                # if dataframe empty
                if attr.empty: continue
                # assign columns
                dfCol = attr.iloc[:, si].rename(nameDict[name])  # get required column
                if requiredDf.empty:
                    requiredDf = dfCol.copy()
                else:
                    requiredDf = pd.concat([requiredDf, dfCol], axis=1)
            ohlcsvs[symbol] = requiredDf
        return ohlcsvs

    # calculate the value of point with respect to that time exchage rate
    def getPointValue(self, point, offset, iloc=True):
        """
        Getting forex point in deposit value (usually calculate the trading cost)
        :param point: float
        :return: {symbol: float}, depend on how many number of symbol (in-deposit)
        """
        pointValues = {}
        for i, symbol in enumerate(self.symbols):
            # getting the quote exchange
            q2d_at = self.quote_exchg.iloc[offset].values[i]
            if not iloc:
                q2d_at = self.quote_exchg.loc[offset].values[i]
            # getting the point values
            pointValues[symbol] = (point * self.all_symbols_info[symbol]['pt_value'] * q2d_at)
        return pointValues

    def getValueDiff(self, offset_s, offset_e, iloc=True):
        """
        Getting the difference of close price between these offsets (In deposit exchange rate)
        :param offset_s:
        :param offset_e:
        :return: {symbol: float}
        """
        pointValueDiffs = {}
        for i, symbol in enumerate(self.symbols):
            # getting the digits
            digits = self.all_symbols_info[symbol]['digits']

            # getting new and old value
            old = self.close[symbol].iloc[offset_s]
            new = self.close[symbol].iloc[offset_e]
            if not iloc:
                old = self.close[symbol].loc[offset_s]
                new = self.close[symbol].loc[offset_e]

            # getting the quote exchange rate
            q2d_at = self.quote_exchg.iloc[offset_e].values[i]
            # calculate the point value difference
            pointValueDiffs[symbol] = (new - old) * (10 ** digits) * self.all_symbols_info[symbol]['pt_value'] * q2d_at
        return pointValueDiffs

    def get_points_dff_df(self, ptValue_need=True, depositValue_need=True):
        """
        :return: get the values in deposit
        """
        # get the digits
        digits = [10 ** self.all_symbols_info[symbol]['digits'] for symbol in self.symbols]

        # get the point values, eg: 1 pt of USDJPY = 100 YEN
        pt_values = [self.all_symbols_info[symbol]['pt_value'] if ptValue_need else 1 for symbol in self.symbols]

        if depositValue_need:
            quote_exchgs = self.quote_exchg.values
        else:
            quote_exchgs = [1 for symbol in self.symbols]

        # get values in deposit
        new_prices = self.close
        old_prices = self.close.shift(periods=1)
        values_diff_df = pd.DataFrame((new_prices - old_prices) * digits * pt_values * quote_exchgs, columns=self.symbols)

        return values_diff_df

    def get_points_dff_df__DISCARD(self, col_names=None, ptValue=True):
        """
        :param col_names: list, set None to use the symbols as column names. Otherwise, rename as fake column name
        :param ptValue: Boolean, if True, return in point value, eg: 1 pt of USDJPY = 100 YEN
        :return: pd.Dataframe: points_dff_values_df
        take the difference from open price
        """
        # if isinstance(self.close, pd.Series):
            # close = pd.DataFrame(self.close, index=self.close.index)  # avoid the error of "too many index" if len(symbols) = 1
        # default ptValue is 1
        pt_value = 1
        # getting the symbols
        symbols = self.close.columns
        # define the prices
        new_prices = self.close
        old_prices = self.close.shift(periods=1)
        points_dff_df = pd.DataFrame(index=new_prices.index)
        for c, symbol in enumerate(symbols):
            digits = self.all_symbols_info[symbol]['digits']  # (note 44b)
            # if need to change the point value
            if ptValue:
                pt_value = self.all_symbols_info[symbol]['pt_value']
            points_dff_df[symbol] = (new_prices.iloc[:, c] - old_prices.iloc[:, c]) * (10 ** digits) * pt_value
        if col_names != None:
            points_dff_df.columns = col_names
        elif col_names == None:
            points_dff_df.columns = symbols
        return points_dff_df



