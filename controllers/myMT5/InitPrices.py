from models.myUtils import dfModel

from dataclasses import dataclass
import pandas as pd


@dataclass
class InitPrices:
    symbols: list
    all_symbols_info: dict
    close: pd.DataFrame
    cc: pd.DataFrame
    ptDv: pd.DataFrame
    quote_exchg: pd.DataFrame
    base_exchg: pd.DataFrame = pd.DataFrame()
    open: pd.DataFrame = pd.DataFrame()
    high: pd.DataFrame = pd.DataFrame()
    low: pd.DataFrame = pd.DataFrame()
    volume: pd.DataFrame = pd.DataFrame()
    spread: pd.DataFrame = pd.DataFrame()

    def getValidCols(self):
        validCol = []
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if not value.empty:
                validCol.append(value)
        return validCol

    def split_Prices(self, percentage):
        trainDict, testDict = {}, {}
        for name, field in self.__dataclass_fields__.items():
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
        nameDict = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume', 'spread': 'spread', 'ptDv': 'ptDv', 'quote_exchg': 'quote_exchg', 'base_exchg': 'base_exchg'}
        for si, symbol in enumerate(self.symbols):
            requiredDf = pd.DataFrame()  # create empty df
            for name, field in self.__dataclass_fields__.items():  # name = variable name; field = pd.dataframe/ value
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
        Getting forex point in deposit value
        :param point: float
        :return: [float], depend on how many number of symbol (in-deposit)
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
        :return:
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


