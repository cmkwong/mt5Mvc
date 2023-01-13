# import sys
# sys.path.append("C:/Users/Chris/projects/210215_mt5/mt5Server")
# sys.path.append("/")

from models.Strategies.SwingScalping.Base import Base

import os
import csv
import numpy as np
import time

class Train(Base):
    def __init__(self, mt5Controller, nodeJsServerController, symbol, startTime, endTime, lot=1):
        super(Train, self).__init__(mt5Controller, nodeJsServerController, symbol)
        self.startTime = startTime
        self.endTime = endTime
        self.LOT = lot
        self.prepare1MinData(startTime, endTime)

    def getSummary(self, masterSignal, ratio_sl_sp, diff_ema_middle_lower, diff_ema_upper_middle, upperEma, middleEma, lowerEma, trendType='rise'):
        count, winRate = self.getWinRate(masterSignal, trendType)
        profit = self.getProfit(masterSignal, trendType)
        summary = {'type': trendType,
                   'count': count,
                   'winRate': winRate,
                   'profit': profit,
                   'ratio_sl_sp': ratio_sl_sp,
                   'diff_ema_middle_lower': diff_ema_middle_lower,
                   'diff_ema_upper_middle': diff_ema_upper_middle,
                   'upperEma': upperEma,
                   'middleEma': middleEma,
                   'lowerEma': lowerEma
                   }
        return summary

    def loopRun(self):
        # define the writer
        r = 0
        # fetch data from database
        fetchData_cust = self.nodeJsServerController.downloadData(self.symbol, self.startTime, self.endTime, timeframe='5min')

        for ratio_sl_sp in np.arange(1.2, 2.2, 0.2):
            for diff_ema_middle_lower in np.arange(20, 80, 10):
                for diff_ema_upper_middle in np.arange(20, 80, 10):
                    # if diff_ema_upper_middle <= 50:
                    #     print('continue middle')
                    #     continue
                    for upperEma in reversed(np.arange(20, 100, 4)):
                        for middleEma in reversed(np.arange(19, upperEma - 1, 4)):
                            for lowerEma in reversed(np.arange(18, middleEma - 1, 4)):
                                # getting master signal
                                start = time.time()
                                masterSignal = self.getMasterSignal(fetchData_cust,
                                                                    lowerEma, middleEma, upperEma,
                                                                    diff_ema_upper_middle, diff_ema_middle_lower,
                                                                    ratio_sl_sp)

                                # build the dictionary to write into csv
                                riseSummary = self.getSummary(masterSignal, ratio_sl_sp, diff_ema_middle_lower, diff_ema_upper_middle, upperEma, middleEma, lowerEma, 'rise')
                                downSummary = self.getSummary(masterSignal, ratio_sl_sp, diff_ema_middle_lower, diff_ema_upper_middle, upperEma, middleEma, lowerEma, 'down')

                                with open(os.path.join(self.backTestDocPath, self.backTestDocName), 'a', newline='', encoding='utf-8') as f:
                                    writer = csv.writer(f)
                                    # write header
                                    if r == 0:
                                        writer.writerow(riseSummary.keys())
                                    # write the rows
                                    writer.writerow(riseSummary.values())
                                    writer.writerow(downSummary.values())
                                    print(riseSummary)
                                    print(downSummary)
                                    r += 2
                                processTime = time.time() - start
                                print(f"Overall Process Time: {processTime}")


# sybmols = ['GBPUSD', 'CADJPY', 'AUDJPY', 'AUDUSD', 'USDCAD', 'USDJPY', 'EURCAD', 'EURUSD']
# mt5Controller = MT5Controller()
# backTest_SwingScalping = Train_SwingScalping(mt5Controller, 'EURUSD', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
# # backTest_SwingScalping.test()
# backTest_SwingScalping.loopRun()
