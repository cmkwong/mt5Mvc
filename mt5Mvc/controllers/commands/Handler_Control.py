
class Handler_Control:
    def __init__(self, mt5Controller, stockPriceLoader):
        self.mt5Controller = mt5Controller
        self.stockPriceLoader = stockPriceLoader

    def run(self, command):
        # switch the nodeJS server env: prod / dev
        if command == '-prod' or command == '-dev':
            self.stockPriceLoader.nodeJsApiController.switch_pro_dev(command[1:])

        # switch the price loader source from mt5 / local
        elif command == '-mt5' or command == '-local' or command == '-sql':
            self.mt5Controller.pricesLoader.switch_source(command[1:])
            self.stockPriceLoader.switch_source(command[1:])

        else:
            return True
