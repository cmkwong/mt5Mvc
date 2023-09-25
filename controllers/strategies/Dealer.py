from models.myUtils import timeModel, printModel

class Dealer:
    def __init__(self, mainController, *, strategy_name, strategy_detail, symbol, timeframe, operation, lot, pt_sl=None, pt_tp=None, exitPoints=None):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsApiController = mainController.nodeJsApiController
        self.openResult = None  # if none, means it has no position yet

        self.strategy_name = strategy_name
        self.strategy_detail = strategy_detail
        self.symbol = symbol
        self.timeframe = timeframe
        self.operation = operation
        self.lot = lot
        self.pt_sl = pt_sl
        self.pt_tp = pt_tp
        self.exitPoints = exitPoints  # {pt: size}, eg: {500: 0.75, 980: 0.25}

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

    def openDeal(self):
        # execute the open position
        request = self.mt5Controller.executor.request_format(symbol=self.symbol,
                                                             operation=self.operation,
                                                             deviation=5,
                                                             lot=self.lot,
                                                             pt_sltp=(self.pt_sl, self.pt_tp))
        # execute request
        self.openResult = self.mt5Controller.executor.request_execute(request)
        if not self.openResult:
            print(f'{self.symbol} open position failed. ')
            self.openResult = None
            return False
        # build the record dictionary
        record = {
            'position_id': self.openResult.order,
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'strategy_detail': self.strategy_detail,
            'operation': self.operation,
            'volume': self.lot,
            'open_price': self.openResult.price, #
            'sl': self.openResult.request.sl,
            'tp': self.openResult.request.tp,
            'open_date': timeModel.getTimeS(outputFormat='%Y-%m-%d'),
            'open_time': timeModel.getTimeS(outputFormat='%H:%M:%S'),
            'exit_points': self._getRecordExitPoints()
        }
        url = self.nodeJsApiController.dealRecordUrl
        self.nodeJsApiController.restRequest(url, {'action': 'open'}, record, 'POST')
        # get and print deal detail
        dealDetail = self.nodeJsApiController.restRequest(url, {'position_id': self.openResult.order})
        printModel.print_dict(dealDetail)

    def closeDeal(self):
        request = self.mt5Controller.executor.close_request_format(self.openResult)
        result = self.mt5Controller.executor.request_execute(request)
        # get the position performance
        profit, swap, commission, duration = self.mt5Controller.get_position_performace(self.openResult.order)
        # get the profit
        # earn = self.mt5Controller.get_position_earn(self.openResult.order)
        # get duration
        # duration = self.mt5Controller.get_position_duration(self.openResult.order)
        record = {
            'position_id': self.openResult.order,
            'close_price': '',
            'swap': swap,
            'commission': commission,
            'profit': profit,
            'duration': duration,
            'finished': 1,
        }
        # save the records into database
        url = self.nodeJsApiController.dealRecordUrl
        self.nodeJsApiController.restRequest(url, {'action': 'close'}, record, 'POST')
        # get and print deal detail
        dealDetail = self.nodeJsApiController.restRequest(url, {'position_id': self.openResult.order})
        printModel.print_dict(dealDetail)
        # set to empty position
        self.openResult = None

    def closeDeal_partial(self, point, size):
        request = self.mt5Controller.executor.close_request_format(self.openResult, size)
        result = self.mt5Controller.executor.request_execute(request)
        # get the position performance
        profit, swap, commission, duration = self.mt5Controller.get_position_performace(self.openResult.order)
        # get the profit
        # earn = self.mt5Controller.get_position_earn(self.openResult.order)
        # get duration
        # duration = self.mt5Controller.get_position_duration(self.openResult.order)
        record = {
            'position_id': self.openResult.order,
            'swap': swap,
            'commission': commission,
            'profit': profit,
            'duration': duration,
            'exit_points': [
                {
                    'tp_position_point': point,
                    'tp_position_size': size,
                    'profit': '', # should be ticket
                    'duration': '', # should be ticket
                    'finished': 1
                }
            ]
        }
        # save the records into database
        url = self.nodeJsApiController.dealRecordUrl
        self.nodeJsApiController.restRequest(url, {'action': 'partial_close'}, record, 'POST')
        # get and print deal detail
        dealDetail = self.nodeJsApiController.restRequest(url, {'position_id': self.openResult.order})
        printModel.print_dict(dealDetail)

    def checkDeal(self):
        # get the position performance
        profit, swap, commission, duration = self.mt5Controller.get_position_performace(self.openResult.order)
        # get duration
        # duration = self.mt5Controller.get_position_duration(self.openResult.order)
        # get the profit
        # earn = self.mt5Controller.get_position_earn(self.openResult.order)
        # build the record dictionary
        record = {
            'position_id': self.openResult.order,
            'profit': profit,
            'swap': swap,
            'commission': commission,
            'duration': duration,
            'finished': 1
        }
        # save the records into database
        url = self.nodeJsApiController.dealRecordUrl
        self.nodeJsApiController.restRequest(url, {'action': 'check'}, record, 'POST')
        # get and print deal detail
        dealDetail = self.nodeJsApiController.restRequest(url, {'position_id': self.openResult.order})
        printModel.print_dict(dealDetail)
        # set to empty position
        self.openResult = None