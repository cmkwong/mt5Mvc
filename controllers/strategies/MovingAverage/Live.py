from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
import time


class Live(Base):
    def __init__(self, mainController):
        self.nodeJsApiController = mainController.nodeJsApiController
        self.mt5Controller = mainController.mt5Controller
        self.openResult = None
        self.lastPositionTime = None
        self.positionsTp = None # for partially close the position (take profit)
        self.digit = None
        self.LOT = 1

    @staticmethod
    def decodeParams(paramDf):

        # create the dict for positions
        param_positions = {}
        positionDf = paramDf.loc[:, ('paramid', 'pt', 'size')]
        for i, row in positionDf.iterrows():
            paramid = row['paramid']
            if paramid not in param_positions.keys():
                param_positions[paramid] = {float(row['pt']): float(row['size'])}
            else:
                param_positions[paramid][float(row['pt'])] = float(row['size'])
        # building the params
        params = {}
        for i, row in paramDf.iterrows():
            paramid = row['id']
            param = row.to_dict()
            if paramid in param_positions.keys():
                param['positions'] = param_positions[paramid]
            params[paramid] = param

        return params

    def getPositionsTp(self, symbol, actionPrice, operation, positions):
        """
        get the same form of position but price as key: {price: size}
        if no position, return empty dictionary
        """
        if not positions:
            return {}
        positionsTp = {}
        for pt, size in positions.items():
            _, tp = self.mt5Controller.executor.transfer_sltp_from_pt(symbol, actionPrice, (0, pt), operation)
            positionsTp[tp] = size
        return positionsTp

    def run(self, *,
            symbol: str = 'USDJPY',
            timeframe: str = '15min',
            fast: int = 5,
            slow: int = 22,
            pt_sl: int = 100,
            pt_tp: int = 210,
            operation: str = 'long',
            lot: int = 1,
            positions: dict = None, # {pt: size}
            **kwargs,
            ):

        while True:
            # check if current position is closed by sl or tp
            if self.openResult and self.mt5Controller.checkOrderClosed(self.openResult) == 0:
                # get the profit
                earn = self.mt5Controller.getPositionEarn(self.openResult)
                print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f} (sltp)')
                self.openResult = None

            # getting the Prices and MaData
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol], count=1000, timeframe=timeframe)
            MaData = self.getMaData(Prices, fast, slow)
            MaData = self.getOperationGroup(MaData)
            # getting curClose and its digit
            curClose = Prices.close[symbol][-1]
            self.digit = Prices.all_symbols_info[symbol]['digits']  # set the digit

            # get the operation group value, either False / datetime
            operationGroupTime = MaData.loc[:, (symbol, f"{operation}_group")][-1]
            # get signal by 'long' or 'short' 1300
            signal = MaData[symbol][operation]

            if not self.openResult:
                if signal.iloc[-1] and not signal.iloc[-2]:
                    # to avoid open the position at same condition
                    if self.lastPositionTime != operationGroupTime:
                        # get the partially position
                        self.positionsTp = self.getPositionsTp(symbol, curClose, operation, positions)
                        # execute the open position
                        request = self.mt5Controller.executor.request_format(symbol=symbol, operation=operation, deviation=5, lot=lot, pt_sltp=(pt_sl, pt_tp))
                        # execute request
                        self.openResult = self.mt5Controller.executor.request_execute(request)
                        # if execute successful
                        if self.mt5Controller.orderSentOk(self.openResult):
                            self.lastPositionTime = signal.index[-1]
                            print(f'{symbol} open position with position id: {self.openResult.order}')
                        else:
                            print(f'{symbol} open position failed. ')
            else:
                # check if signal should be close
                if not signal.iloc[-1] and signal.iloc[-2]:
                    request = self.mt5Controller.executor.close_request_format(self.openResult)
                    result = self.mt5Controller.executor.request_execute(request)
                    # get the profit
                    earn = self.mt5Controller.getPositionEarn(self.openResult)
                    # print(f'{symbol} close position with position id: {result.request.order}')
                    print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f} (By Signal Close)')
                    self.openResult = None

                # check if the partially position being reached
                for position, size in self.positionsTp.items():
                    # check if available size left
                    if size > 0:
                        # calculate the stop loss and take profit
                        if curClose >= position:
                            request = self.mt5Controller.executor.close_request_format(self.openResult, size)
                            result = self.mt5Controller.executor.request_execute(request)
                            # get the profit
                            earn = self.mt5Controller.getPositionEarn(self.openResult)
                            # print(f'{symbol} close position with position id: {result.request.order}')
                            print(f'{symbol} position closed with position id: {self.openResult.order} and earn: {earn:2f} (Partial)')
                            # reset the position
                            self.positionsTp[position] = 0

            # delay the operation
            time.sleep(5)
