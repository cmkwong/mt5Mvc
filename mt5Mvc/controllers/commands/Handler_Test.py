from mt5Mvc.controllers.strategies.Dealer import Dealer
from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.controllers.myStock.StockPriceLoader import StockPriceLoader
from mt5Mvc.controllers.strategies.MovingAverage.Train import Train as MovingAverageTrain
from mt5Mvc.controllers.strategies.MovingAverage.Backtest import Backtest as MovingAverageBacktest

from mt5Mvc.models.myUtils import paramModel, timeModel

class Handler_Test:
    def __init__(self):
        self.mt5Controller = MT5Controller()
        self.stockPriceLoader = StockPriceLoader()
        self.movingAverageTrainer = MovingAverageTrain()
        self.movingAverageBacktest = MovingAverageBacktest()

    def run(self, command):
        # testing for getting the data from sql / mt5 by switch the data source
        if command == '-testPeriod':
            self.mt5Controller.pricesLoader.getPrices(
                symbols=['USDJPY'],
                start=(2023, 2, 18, 0, 0),
                end=(2023, 7, 20, 0, 0),
                timeframe='1min'
            )

        # testing for getting current data from sql / mt5 by switch the data source
        elif command == '-testCurrent':
            self.mt5Controller.pricesLoader.getPrices(
                symbols=['USDJPY'],
                count=1000,
                timeframe='15min'
            )

        elif command == '-testDeal':
            # define the dealer
            dealer = Dealer(strategy_name='Test',
                            strategy_detail='Test_detail',
                            symbol='USDJPY',
                            timeframe='15min',
                            operation='long',
                            lot=0.1,
                            pt_sl=500,
                            exitPoints={900: 0.75, 1200: 0.2, 1500: 0.05}
                            )
            dealer.openDeal()
            dealer.closeDeal()

        elif command == '-testMt5':
            historicalOrder = self.mt5Controller.get_historical_order(lastDays=3, position_id=338232986)
            historicalDeals = self.mt5Controller.get_historical_deals(lastDays=3, position_id=338232986)

        elif command == '-readLocal':
            params = paramModel.ask_param(
                {
                    'path': ['./docs/us_historical_data', str],
                    'timeframe': ['1D', str]
                 }
            )
            # self.stockPriceLoader.switch_source('local')
            Prices = self.stockPriceLoader.getPrices_from_local(**params)
            self.movingAverageTrainer.analysis(Prices, 250)

        elif command == '-testimg':
            params = paramModel.ask_param(
                {
                    'path': ['./docs/us_historical_data', str],
                    'timeframe': ['1D', str]
                }
            )
            # self.stockPriceLoader.switch_source('local')
            Prices = self.stockPriceLoader.getPrices_from_local(**params)
            self.movingAverageBacktest.getMaDistImg(Prices, 176, 179, 'long')

        else:
            return True
