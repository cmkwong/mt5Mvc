import sys

sys.path.append("C:/Users/Chris/projects/210215_mt5/mt5Server")
sys.path.append("/")

from myDataFeed.myMt5.MT5Controller import MT5Controller
from models.Strategies.SwingScalping.Base_SwingScalping import Base_SwingScalping


class Backtest_SwingScalping(Base_SwingScalping):
    def __init__(self, mt5Controller, nodeJsServerController, *, symbol):
        super(Backtest_SwingScalping, self).__init__(mt5Controller, nodeJsServerController, symbol)

    def run(self, startTime, endTime, lot=1, diff_ema_upper_middle=45, diff_ema_middle_lower=30, ratio_sl_sp=1.5,
            lowerEma=25, middleEma=50, upperEma=100):
        self.startTime = startTime
        self.endTime = endTime
        self.lot = lot
        self.prepare1MinData(startTime, endTime)
        fetchData_cust = self.nodeJsServerController.downloadData(self.symbol, startTime, endTime, timeframe='5min')

        masterSignal = self.getMasterSignal(fetchData_cust,
                                            lowerEma, middleEma, upperEma,
                                            diff_ema_upper_middle, diff_ema_middle_lower,
                                            ratio_sl_sp)

        return masterSignal


# mt5Controller = MT5Controller()
# backtest_SwingScalping = Backtest_SwingScalping(mt5Controller, 'USDJPY', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
# masterSignal = backtest_SwingScalping.run(diff_ema_upper_middle=45, diff_ema_middle_lower=30, ratio_sl_sp=1.5,
#                            lowerEma=25, middleEma=50, upperEma=100)
