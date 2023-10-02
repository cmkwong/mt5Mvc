from models.myUtils import timeModel, printModel

class Dealer:
    def __init__(self, mainController, *,
                 symbol, timeframe, operation, lot,
                 pt_sl=None, pt_tp=None, exitPoints=None,
                 strategy_name='', strategy_id='', strategy_detail=''):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsApiController = mainController.nodeJsApiController
        self.openResult = None  # if none, means it has no position yet

        # params
        self.symbol = symbol
        self.timeframe = timeframe
        self.operation = operation
        self.lot = lot
        self.pt_sl = pt_sl
        self.pt_tp = pt_tp
        self.exitPoints = exitPoints  # {pt: size}, eg: {500: 0.75, 980: 0.25}
        # strategy info
        self.strategy_name = strategy_name
        self.strategy_id = strategy_id
        self.strategy_detail = strategy_detail

    def getExitPrices_tp(self, actionPrice):
        """
        get the same form of position but price as key: {point: size} -> {price: {point, size}}
        if no position, return empty dictionary
        """
        if not self.exitPoints:
            return {}
        exitPrices = {}
        for pt, size in self.exitPoints.items():
            _, tp = self.mt5Controller.executor.transfer_sltp_from_pt(self.symbol, actionPrice, (0, pt), self.operation)
            exitPrices[tp] = {'point': pt, 'size': size}
        return exitPrices

    def _getRecordExitPoints(self):
        exitPoints = []
        for pt, size in self.exitPoints.items():
            exitPoints.append({'tp_position_point': pt, 'tp_position_size': size})
        return exitPoints

    def update_deal(self):
        # build the record dictionary
        records = self.mt5Controller.get_historical_deals(position_id=self.openResult.order, datatype=dict)
        url = self.nodeJsApiController.dealRecordUrl
        for record in records:
            # append self-defined fields
            record['strategy_name'] = self.strategy_name
            record['strategy_id'] = self.strategy_id
            self.nodeJsApiController.restRequest(url, None, record, 'POST')
        # print the record being uploaded
        printModel.print_dict(records, True, orient='columns')

    def openDeal(self, comment=''):
        # execute the open position
        request = self.mt5Controller.executor.request_format(symbol=self.symbol,
                                                             operation=self.operation,
                                                             deviation=5,
                                                             lot=self.lot,
                                                             pt_sltp=(self.pt_sl, self.pt_tp),
                                                             comment=comment)
        # execute request
        self.openResult = self.mt5Controller.executor.request_execute(request)
        if not self.openResult:
            print(f'{self.symbol} open position failed. ')
            self.openResult = None
            return False
        # update into NodeJS
        self.update_deal()

    def closeDeal(self, comment=''):
        # if no open positions, then return false
        if not self.openResult:
            return False
        request = self.mt5Controller.executor.close_request_format(self.openResult, comment=comment)
        result = self.mt5Controller.executor.request_execute(request)
        # update into NodeJS
        self.update_deal()
        # set to empty position
        self.openResult = None

    def closeDeal_partial(self, size, comment='Partial Close'):
        request = self.mt5Controller.executor.close_request_format(self.openResult, size, comment=comment)
        result = self.mt5Controller.executor.request_execute(request)
        # update into NodeJS
        self.update_deal()

    def checkDeal(self):
        # update into NodeJS
        self.update_deal()