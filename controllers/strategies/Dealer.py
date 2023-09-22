from models.myUtils import timeModel

class Dealer:
    def __init__(self, mainController, *, strategy_name, strategy_detail, symbol, timeframe, operation, lot, pt_sl=None, pt_tp=None):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsApiController = mainController.nodeJsApiController

        self.strategy_name = strategy_name
        self.strategy_detail = strategy_detail
        self.symbol = symbol
        self.timeframe = timeframe
        self.operation = operation
        self.lot = lot
        self.pt_sl = pt_sl
        self.pt_tp = pt_tp

    def openPosition(self):
        # execute the open position
        request = self.mt5Controller.executor.request_format(symbol=self.symbol,
                                                             operation=self.operation,
                                                             deviation=5,
                                                             lot=self.lot,
                                                             pt_sltp=(self.pt_sl, self.pt_tp))
        # execute request
        self.openResult = self.mt5Controller.executor.request_execute(request)
        sample = {
            'position_id': self.openResult.order,
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'strategy_detail': self.strategy_detail,
            'type': self.operation,
            'volume': self.lot,
            'open_price': self.openResult.price, #
            'sl': self.openResult.request.sl,
            'tp': self.openResult.request.tp,
            'open_date': timeModel.getTimeS(outputFormat='%Y-%m-%d'),
            'open_time': timeModel.getTimeS(outputFormat='%H:%M:%S'),
        }

    def closePosition(self):
        pass

    def partialClosePosition(self):
        pass

    def checkClosed(self):
        pass