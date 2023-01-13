# import sys
#
# sys.path.append("C:/Users/Chris/projects/210215_mt5/mt5Server")
# sys.path.append("/")

from myDataFeed.myMt5.MT5Controller import MT5Controller
from models.Strategies.SwingScalping.Base import Base


class Backtest(Base):
    def __init__(self, mt5Controller, nodeJsServerController, *, symbol: str):
        super(Backtest, self).__init__(mt5Controller, nodeJsServerController, symbol)

    def run(self, *, startTime: tuple, endTime: tuple, lot: int, diff_ema_upper_middle: int, diff_ema_middle_lower: int, ratio_sl_sp: float,
            lowerEma: int, middleEma: int, upperEma: int):
        self.startTime = startTime
        self.endTime = endTime
        self.lot = lot
        self.prepare1MinData(startTime, endTime)
        fetchData_cust = self.nodeJsServerController.downloadData(self.symbol, startTime, endTime, timeframe='5min')

        # getting master signal
        masterSignal = self.getMasterSignal(fetchData_cust,
                                            lowerEma, middleEma, upperEma,
                                            diff_ema_upper_middle, diff_ema_middle_lower,
                                            ratio_sl_sp)
        # printing the results
        for trendType in ['rise', 'down']:
            count, winRate = self.getWinRate(masterSignal, trendType)
            profit = self.getProfit(masterSignal, trendType)
            print(f"Type: {trendType}, Count: {count}, Win Rate: {winRate}, Profit: {profit}")

# mt5Controller = MT5Controller()
# backtest_SwingScalping = Backtest_SwingScalping(mt5Controller, 'USDJPY', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
# masterSignal = backtest_SwingScalping.run(diff_ema_upper_middle=45, diff_ema_middle_lower=30, ratio_sl_sp=1.5,
#                            lowerEma=25, middleEma=50, upperEma=100)
