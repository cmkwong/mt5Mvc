import os
from controllers.strategies.SwingScalping.Base import Base

class Backtest(Base):
    def __init__(self, mt5Controller, nodeJsServerController):
        super(Backtest, self).__init__(mt5Controller, nodeJsServerController)

    @property
    def getName(self):
        parentFolder = os.path.basename(os.getcwd())
        return f'{parentFolder}({self.__class__.__name__})'

    def run(self, *, symbol: str, startTime: tuple, endTime: tuple, lot: int, diff_ema_upper_middle: int, diff_ema_middle_lower: int, ratio_sl_sp: float,
            lowerEma: int, middleEma: int, upperEma: int):
        self.startTime = startTime
        self.endTime = endTime
        self.lot = lot
        self.prepare1MinData(symbol, startTime, endTime)
        fetchData_cust = self.nodeJsServerController.downloadSymbolData(symbol, startTime, endTime, timeframe='5min')

        # getting master signal
        masterSignal = self.getMasterSignal(symbol, fetchData_cust,
                                            lowerEma, middleEma, upperEma,
                                            diff_ema_upper_middle, diff_ema_middle_lower,
                                            ratio_sl_sp)
        # printing the results
        for trendType in ['rise', 'down']:
            count, winRate = self.getWinRate(masterSignal, trendType)
            profit = self.getProfit(masterSignal, trendType)
            print(f"Type: {trendType}, Count: {count}, Win Rate: {winRate}, Profit: {profit}")
