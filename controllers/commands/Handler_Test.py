from controllers.strategies.Dealer import Dealer

class Handler_Test:
    def __init__(self, nodeJsApiController, mt5Controller, stockPriceLoader, threadController, strategyController, plotController):
        self.nodeJsApiController = nodeJsApiController
        self.mt5Controller = mt5Controller
        self.stockPriceLoader = stockPriceLoader
        self.threadController = threadController
        self.strategyController = strategyController
        self.plotController = plotController

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
            dealer = Dealer(self.mt5Controller, self.nodeJsApiController,
                            strategy_name='Test',
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

        else:
            return True
