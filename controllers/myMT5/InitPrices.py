from models.myUtils import dfModel

from dataclasses import dataclass
import pandas as pd

@dataclass
class InitPrices:
    symbols: list
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
