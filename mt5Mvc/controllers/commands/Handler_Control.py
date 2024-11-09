from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.controllers.myStock.StockPriceLoader import StockPriceLoader

class Handler_Control:
    def __init__(self):
        self.mt5Controller = MT5Controller()
        self.stockPriceLoader = StockPriceLoader()

    def run(self, command):
        # switch the nodeJS server env: prod / dev
        if command == '-prod' or command == '-dev':
            self.stockPriceLoader.nodeJsApiController.switch_pro_dev(command[1:])

        # switch the price loader source from mt5 / local
        elif command == '-mt5' or command == '-local' or command == '-sql':
            self.mt5Controller.pricesLoader.switch_source(command[1:])
            self.stockPriceLoader.switch_source(command[1:])

        elif command == '-start':
            self.mt5Controller.connect_server()

        else:
            return True
