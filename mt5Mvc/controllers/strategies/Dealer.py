from mt5Mvc.models.myUtils import timeModel, printModel
from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.controllers.myNodeJs.NodeJsApiController import NodeJsApiController
class Dealer:
    def __init__(self, symbol, timeframe, operation, lot,
                 pt_sl=None, pt_tp=None, exitPoints=None,
                 strategy_name='', strategy_id='', strategy_detail=''):
        self.mt5Controller = MT5Controller()
        self.nodeJsApiController = NodeJsApiController()
        self.position_id = None # if none, means it has no position yet

        # params
        self.symbol = symbol
        self.timeframe = timeframe
        self.operation = operation
        self.lot = lot
        self.pt_sl = pt_sl
        self.pt_tp = pt_tp
        self.exitPoints = exitPoints  # [ [exit_id ,point, size ], ... ], eg: [ [ 12, 500, 0.75], [13, 980, 0.25], ... ]
        # strategy data
        self.strategy_name = strategy_name
        self.strategy_id = strategy_id
        self.strategy_detail = strategy_detail

    def getExitPrices_tp(self, actionPrice):
        """
        get the same form of position but price as key: { price: {exit_id, point, size} }
        if no position, return empty dictionary
        """
        if not self.exitPoints:
            return {}
        self.exitPrices = {}
        for exit_id, pt, size in self.exitPoints:
            _, tp = self.mt5Controller.executor.transfer_sltp_from_pt(self.symbol, actionPrice, (0, pt), self.operation)
            self.exitPrices[tp] = {'exit_id': exit_id, 'point': pt, 'size': size} # tp is price format

    # def _getRecordExitPoints(self):
    #     exitPoints = []
    #     for pt, size in self.exitPoints.items():
    #         exitPoints.append({'tp_position_point': pt, 'tp_position_size': size})
    #     return exitPoints

    def update_deal(self, info: dict = None):
        # build the record dictionary (only update last deal)
        record = self.mt5Controller.get_historical_deals(position_id=self.position_id, datatype=dict)[-1]
        url = self.nodeJsApiController.dealRecordUrl

        # append self-defined fields
        record['strategy_name'] = self.strategy_name
        record['strategy_id'] = self.strategy_id
        # merge user-defined and pre-defined data into only last deal
        if info: record.update(info)
        # send request
        self.nodeJsApiController.restRequest(url, None, record, 'POST')
        # print the record being uploaded
        printModel.print_dict(record, True, orient='index')

    def openDeal(self, info: dict = None, comment=''):
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
            self.position_id = None
            return False
        # assign the position id
        self.position_id = self.openResult.order
        # update into NodeJS
        self.update_deal(info)

        return True

    def closeDeal(self, info: dict = None, comment=''):
        # if no open positions, then return false
        if not self.position_id:
            return False
        request = self.mt5Controller.executor.close_request_format(position_id=self.position_id, comment=comment)
        result = self.mt5Controller.executor.request_execute(request)
        # update into NodeJS, if succeed
        if result:
            self.update_deal(info)
            # set to empty position
            self.position_id = None
        return result

    def closeDeal_partial(self, size, info: dict = None, comment: str = 'Partial Close'):
        request = self.mt5Controller.executor.close_request_format(position_id=self.position_id, percent=size, comment=comment)
        result = self.mt5Controller.executor.request_execute(request)
        # update into NodeJS, if succeed
        if result:
            self.update_deal(info)
        return result

    def checkDeal(self):
        # update into NodeJS
        self.update_deal()
        return True
