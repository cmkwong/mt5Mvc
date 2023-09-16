from controllers.myMT5.MT5PricesLoader import MT5PricesLoader
from controllers.myMT5.MT5Executor import MT5Executor
from controllers.myMT5.MT5SymbolController import MT5SymbolController
from controllers.myMT5.MT5TickController import MT5TickController
from controllers.myMT5.MT5TimeController import MT5TimeController
from models.myUtils import dfModel

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, time
import pytz
import config


class MT5Controller:
    def __init__(self, nodeJsApiController):
        self.connect_server()
        self.symbolController = MT5SymbolController()
        self.tickController = MT5TickController()
        self.timeController = MT5TimeController()
        self.pricesLoader = MT5PricesLoader(self.timeController, self.symbolController, nodeJsApiController)  # loading the loader
        self.executor = MT5Executor(self.pricesLoader.all_symbols_info)  # execute the request (buy/sell)

    def print_terminal_info(self):
        # request connection status and parameters
        print(mt5.terminal_info())
        # request account info
        print(mt5.account_info())
        # get loader on MetaTrader 5 version
        print(mt5.version())

    def print_active_positions(self):
        """
        :return: print all of the active order situation
        """
        cols = ['ticket', 'time', 'volume', 'type', 'profit', 'price_open', 'price_current']
        postions = mt5.positions_get()
        datas = {}
        for i, position in enumerate(postions):
            datas[i+1] = []
            for col in cols:
                v = getattr(position, col)
                if col == 'time':
                    v = datetime.fromtimestamp(v) + timedelta(hours=-8)
                datas[i + 1].append(v)
        positionsDf = pd.DataFrame.from_dict(datas, orient='index', columns=cols)
        dfModel.printDf(positionsDf)

    def print_historical_deals(self, *, lastDays: int = 1):
        deals = self.getHistoricalDeals(lastDays=lastDays)
        dfModel.printDf(deals)

    def getHistoricalDeals(self, *, lastDays: int = 365):
        """
        :lastDays: 1625 = 5 years
        :return pd.Dataframe of deals
        """
        cols = ['time', 'order', 'type', 'position_id', 'reason', 'volume', 'price', 'commission', 'swap', 'profit', 'fee', 'symbol']
        currentDate = datetime.today() + timedelta(hours=8)  # time object
        # currentDate = pytz.timezone('Hongkong').localize(datetime.today())
        fromDate = currentDate - timedelta(days=lastDays)
        deals = mt5.history_deals_get(fromDate, currentDate)
        datas = {}
        for deal in deals:
            # time = datetime.fromtimestamp(deal.time)
            row = []
            for col in cols:
                row.append(getattr(deal, col))
            datas[deal.ticket] = row
        historicalDeals = pd.DataFrame.from_dict(datas, orient='index', columns=cols)
        # transfer seconds into time
        historicalDeals['time'] = historicalDeals['time'].apply(datetime.fromtimestamp)
        # resume back to UTC time
        historicalDeals['time'] = historicalDeals['time'].apply(lambda t: t + timedelta(hours=-8))
        return historicalDeals

    def getPositionEarn(self, openResult):
        positionId = openResult.order
        # get required deal in last 1 year
        historicalDeals = self.getHistoricalDeals()
        # sum all of the profit with same position id
        profits = historicalDeals.groupby('position_id')['profit'].sum()
        if positionId not in profits.index:
            print(f"The required Position Id: {positionId} is not existed. ")
        return profits[positionId]

    def get_historical_order(self, lastDays: int = 10):
        """
        :return:
        """
        currentDate = datetime.today()  # time object
        fromDate = currentDate - timedelta(days=lastDays)
        historicalOrder = mt5.history_orders_get(fromDate, currentDate)
        return historicalOrder

    def getPositionDuration(self, openResult):
        """
        get the duration in time format (00: 00: 00)
        :param strin
        """
        # get all the positions with same position id
        positionId = openResult.order  # ticket ID, in metatrader position ID is same as ticket ID
        positions = mt5.history_orders_get(position=positionId)
        durations = []
        for position in positions:
            # append the duration
            durations.append(position.time_done)
        # calculate the duration taken
        seconds = max(durations) - min(durations)
        # duration = time(hour=seconds // 3600, minute=(seconds  - (seconds % 60)) // 60 % 60, second=seconds % 60)
        duration = timedelta(seconds=seconds)
        return duration

    def checkOrderClosed(self, openResult):
        """
        Check if order finished or not
        :param openResult: the result after execute the request in mt5
        :return: Boolean
        """
        # get all the positions with same position id
        positionId = openResult.order  # ticket ID, in metatrader position ID is same as ticket ID
        positions = mt5.history_orders_get(position=positionId)
        # order finished return True, otherwise, False
        volumeBalance = 0.0
        for position in positions:
            factor = 1 if position.type == 1 else -1
            volumeBalance += factor * position.volume_initial
        return volumeBalance

    def orderSentOk(self, result):
        """
        :param result: Check if mt5 order sent and return successful code
        """
        if result and (result.retcode == mt5.TRADE_RETCODE_DONE):
            return True
        print(f"Order Not OK: \n{result}")
        return False

    def connect_server(self):
        # connect to MetaTrader 5
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        else:
            print("Connecting MetaTrader 5 ... ")

    def disconnect_server(self):
        # disconnect to MetaTrader 5
        mt5.shutdown()
        print("MetaTrader Shutdown.")

