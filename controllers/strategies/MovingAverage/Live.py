from models.myUtils.paramModel import SymbolList, DatetimeTuple
from models.myUtils import printModel
from controllers.strategies.MovingAverage.Base import Base
from controllers.strategies.Dealer import Dealer
import time


class Live(Base, Dealer):
    def __init__(self, mainController, *, symbol: str = 'USDJPY', timeframe: str = '15min', fast: int = 5, slow: int = 22, pt_sl: int = 100, pt_tp: int = 210, operation: str = 'long', lot: int = 1, exitPoints: dict = None, **kwargs):
        super(Live, self).__init__(mainController, strategy_name='Simple MA', strategy_detail=f'{fast}/{slow}', symbol=symbol, timeframe=timeframe, operation=operation, lot=lot, pt_sl=pt_sl, pt_tp=pt_tp, exitPoints=exitPoints)
        self.nodeJsApiController = mainController.nodeJsApiController
        self.mt5Controller = mainController.mt5Controller
        self.openResult = None
        self.lastPositionTime = None
        self.exitPrices = None  # for partially close the position (take profit)
        self.digit = None

        # for records
        self.strategy_name = 'Simple MA'
        self.strategy_detail = f'{fast}/{slow}'

        # run parameters
        self.fast = fast
        self.slow = slow
        # self.symbol = symbol
        # self.timeframe = timeframe
        # self.pt_sl = pt_sl
        # self.pt_tp = pt_tp
        # self.operation = operation
        # self.lot = lot
        # self.positions = positions  # {pt: size}

    @staticmethod
    def decodeParams(paramDf):

        # create the dict for positions
        param_positions = {}
        positionDf = paramDf.loc[:, ('paramid', 'pt', 'size')]
        for i, row in positionDf.iterrows():
            if not row['paramid']:
                continue
            paramid = row['paramid']
            if paramid not in param_positions.keys():
                param_positions[paramid] = {float(row['pt']): float(row['size'])}
            else:
                param_positions[paramid][float(row['pt'])] = float(row['size'])
        # building the params
        params = {}
        for i, row in paramDf.iterrows():
            paramid = row['id']
            # change dataframe row into dictionary
            param = row.to_dict()
            if paramid in param_positions.keys():
                param['exitPoints'] = param_positions[paramid]
            params[paramid] = param

        return params

    def run(self):

        while True:
            # check if current position is closed by sl or tp
            if self.openResult and self.mt5Controller.checkOrderClosed(self.openResult) == 0:
                # get duration
                duration = self.mt5Controller.getPositionDuration(self.openResult)
                # get the profit
                earn = self.mt5Controller.getPositionEarn(self.openResult)
                printModel.print_dict({
                    'Symbol': self.symbol,
                    'Strategy': f'{self.fast}/{self.slow} sl: {self.openResult.request.sl} tp: {self.openResult.request.tp}',
                    'Operation': 'closed',
                    'Reason': 'sltp',
                    'Position ID': self.openResult.order,
                    'Balance': f'{earn:2f}',
                    'Duration': duration,
                    'Remark': ''
                }, True)
                # print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f} (sltp) and time taken {duration}')
                self.openResult = None

            # getting the Prices and MaData
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[self.symbol], count=1000, timeframe=self.timeframe)
            MaData = self.getMaData(Prices, self.fast, self.slow)
            MaData = self.getOperationGroup(MaData)
            # getting curClose and its digit
            curClose = Prices.close[self.symbol][-1]
            self.digit = Prices.all_symbols_info[self.symbol]['digits']  # set the digit

            # get the operation group value, either False / datetime
            operationGroupTime = MaData.loc[:, (self.symbol, f"{self.operation}_group")][-1]
            # get signal by 'long' or 'short' 1300
            signal = MaData[self.symbol][self.operation]

            if not self.openResult:
                # open position signal
                if signal.iloc[-1] and not signal.iloc[-2]:
                    # to avoid open the position at same condition
                    if self.lastPositionTime != operationGroupTime:
                        # get the partially position
                        self.exitPrices = self.getExitPrices_tp(curClose)
                        # execute the open position
                        request = self.mt5Controller.executor.request_format(symbol=self.symbol, operation=self.operation, deviation=5, lot=self.lot, pt_sltp=(self.pt_sl, self.pt_tp))
                        # execute request
                        self.openResult = self.mt5Controller.executor.request_execute(request)
                        # if execute successful
                        if self.mt5Controller.orderSentOk(self.openResult):
                            self.lastPositionTime = signal.index[-1]
                            printModel.print_dict({
                                'Symbol': self.symbol,
                                'Strategy': f'{self.fast}/{self.slow} sl: {self.openResult.request.sl} tp: {self.openResult.request.tp}',
                                'Operation': 'opened',
                                'Reason': '',
                                'Position ID': self.openResult.order,
                                'Balance': 0,
                                'Duration': 0,
                                'Remark': ''
                            }, True)
                            # print(f"{symbol} open position with position id: {self.openResult.order}; {fast}/{slow} sl: {self.openResult.request.sl} tp: {self.openResult.request.tp}")
                        else:
                            print(f'{self.symbol} open position failed. ')
            else:
                # check if signal should be close
                if not signal.iloc[-1] and signal.iloc[-2]:
                    request = self.mt5Controller.executor.close_request_format(self.openResult)
                    result = self.mt5Controller.executor.request_execute(request)
                    # get the profit
                    earn = self.mt5Controller.getPositionEarn(self.openResult)
                    # get duration
                    duration = self.mt5Controller.getPositionDuration(self.openResult)
                    printModel.print_dict({
                        'Symbol': self.symbol,
                        'Strategy': f'{self.fast}/{self.slow} sl: {self.openResult.request.sl} tp: {self.openResult.request.tp}',
                        'Operation': 'closed',
                        'Reason': 'By Signal Close',
                        'Position ID': self.openResult.order,
                        'Balance': f'{earn:2f}',
                        'Duration': duration,
                        'Remark': ''
                    }, True)
                    # print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f} (By Signal Close) and time taken {duration}')
                    self.openResult = None

                # check if the partially position being reached
                for exitPrice, data in self.exitPrices.items():
                    size = data['size']
                    point = data['point']
                    # check if available size left
                    if size > 0:
                        # calculate the stop loss and take profit
                        if curClose >= exitPrice:
                            request = self.mt5Controller.executor.close_request_format(self.openResult, size)
                            result = self.mt5Controller.executor.request_execute(request)
                            # get the profit
                            earn = self.mt5Controller.getPositionEarn(self.openResult)
                            # get duration
                            duration = self.mt5Controller.getPositionDuration(self.openResult)
                            printModel.print_dict({
                                'Symbol': self.symbol,
                                'Strategy': f'{self.fast}/{self.slow} sl: {self.openResult.request.sl} tp: {self.openResult.request.tp}',
                                'Operation': 'closed',
                                'Reason': 'Partially',
                                'Position ID': self.openResult.order,
                                'Balance': f'{earn:2f}',
                                'Duration': duration,
                                'Remark': ''
                            }, True)
                            # print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f} (Partial) and time taken {duration}')
                            # reset the position
                            self.exitPrices[exitPrice] = 0

            # delay the operation
            time.sleep(5)
