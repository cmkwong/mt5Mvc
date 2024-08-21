from controllers.strategies.SwingScalping.Base import Base
from models.myUtils.printModel import print_at
import os
import time

class Live(Base):
    def __init__(self, mainController, *, auto=False):
        super(Live, self).__init__(mainController.mt5Controller, mainController.nodeJsApiController)
        # init the variables
        self.breakThroughTime = None
        self.breakThroughCondition, self.trendRangeCondition = False, False
        # check if notice have been announced
        self.inPosition = False
        # auto
        self.auto = auto
        self.tg = False

    @property
    def getIdentity(self):
        return f"{self.__class__.__name__}: {self.trendType} {self.diff_ema_upper_middle} {self.diff_ema_middle_lower} {self.lowerEma} {self.middleEma} {self.upperEma} {self.ratio_sl_sp} {self.lot}"

    def __set_name__(self, owner, name):
        parentFolder = os.path.basename(os.getcwd())
        return f'{parentFolder}({self.__class__.__name__})'

    def checkValidAction(self, symbol, masterSignal, trendType='rise'):
        lastRow = masterSignal.iloc[-1, :]
        currentIndex = masterSignal.index[-1]
        # meet the conditions and have not trade that before and not in-position
        if (lastRow[trendType + 'Break'] and lastRow[trendType + 'Range'] and currentIndex != self.breakThroughTime and not self.inPosition):
            print(f'{symbol} {self.trendType} action going ...\n')
            self.status = {'type': trendType, 'time': self.breakThroughTime, 'sl': lastRow['stopLoss'], 'tp': lastRow['takeProfit']}
            print(self.status)
            # make notice and take action if auto set to True
            statusTxt = f'{symbol}\n'
            for k, v in self.status.items():
                statusTxt += f"{k} {v}\n"
            if not self.auto:
                print(statusTxt)
            elif self.auto:
                # define the action type
                if self.status['type'] == 'rise':
                    actionType = 'long'
                else:
                    actionType = 'short'
                # build request format
                request = self.mt5Controller.executor.request_format(symbol=symbol, operation=actionType, deviation=5, lot=self.lot, sltp=(float(self.status['sl']), float(self.status['tp'])))
                # execute request
                self.mt5Controller.executor.request_execute(request)
                self.breakThroughTime = masterSignal.index[-1]  # save the last toke action time
                self.inPosition = True
        # reset the notice if in next time slot
        if self.inPosition:
            lastPrice = masterSignal.iloc[-1]['close']
            if (self.breakThroughTime != masterSignal.index[-1]):  # not at the same time
                if (self.trendType == 'rise' and (lastPrice >= self.status['tp'] or lastPrice <= self.status['sl'])) or (self.trendType == 'down' and (lastPrice <= self.status['tp'] or lastPrice >= self.status['sl'])):
                    self.inPosition = False

    def run(self, *,
            symbol: str,
            diff_ema_upper_middle: int,
            diff_ema_middle_lower: int,
            ratio_sl_sp: float,
            lowerEma: int,
            middleEma: int,
            upperEma: int,
            trendType: str,
            lot: int):
        self.diff_ema_upper_middle = diff_ema_upper_middle
        self.diff_ema_middle_lower = diff_ema_middle_lower
        self.ratio_sl_sp = ratio_sl_sp
        self.lowerEma = lowerEma
        self.middleEma = middleEma
        self.upperEma = upperEma
        self.trendType = trendType  # 'rise' / 'down'
        self.lot = lot

        while True:
            time.sleep(5)
            # getting latest Prices
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol],
                                                               start=None,
                                                               end=None,
                                                               timeframe='1H',
                                                               count=1000,
                                                               ohlcvs='111111'
                                                               )
            # getting ohlcvs
            ohlcvs = Prices.get_ohlcvs_from_prices()[symbol]

            # getting master signal
            masterSignal = self.getMasterSignal(symbol, ohlcvs, self.lowerEma, self.middleEma, self.upperEma,
                                                self.diff_ema_upper_middle, self.diff_ema_middle_lower,
                                                self.ratio_sl_sp, needEarning=False)

            self.checkValidAction(symbol, masterSignal, self.trendType)

# get live Data from MT5 Server
# mt5Controller = Mt5Controller()
# swingScalping = SwingScalping(mt5Controller, 'USDJPY', breakThroughCondition='50')
# swingScalping.run()
