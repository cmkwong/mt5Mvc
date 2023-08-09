from controllers.myMT5.MT5PricesLoader import MT5PricesLoader
from controllers.myMT5.MT5Executor import MT5Executor
from controllers.myMT5.MT5SymbolController import MT5SymbolController
from controllers.myMT5.MT5TickController import MT5TickController
from controllers.myMT5.MT5TimeController import MT5TimeController

import MetaTrader5 as mt5
from datetime import datetime, timedelta
import config

class MT5Controller:
    def __init__(self, nodeJsApiController):
        self.connect_server()
        self.symbolController = MT5SymbolController()
        self.tickController = MT5TickController()
        self.timeController = MT5TimeController()
        self.executor = MT5Executor()  # execute the request (buy/sell)
        self.pricesLoader = MT5PricesLoader(self.timeController, self.symbolController, nodeJsApiController)  # loading the loader

    def print_terminal_info(self):
        # request connection status and parameters
        print(mt5.terminal_info())
        # request account info
        print(mt5.account_info())
        # get loader on MetaTrader 5 version
        print(mt5.version())

    def get_historical_deal(self, lastDays=10):
        """
        :return:
        """
        currentDate = datetime.today() # time object
        fromDate = currentDate - timedelta(days=lastDays)
        historicalDeal = mt5.history_deals_get(fromDate, currentDate)
        return historicalDeal

    def get_historical_order(self, lastDays=10):
        """
        :return:
        """
        currentDate = datetime.today() # time object
        fromDate = currentDate - timedelta(days=lastDays)
        historicalOrder = mt5.history_orders_get(fromDate, currentDate)
        return historicalOrder

    def orderFinished(self, positionId):
        """
        Check if order finished or not
        :param positionId: ticket ID, in metatrader position ID is same as ticket ID
        :return: Boolean
        """
        # get all the positions with same position id
        positions = mt5.history_orders_get(position=positionId)
        # order finished return True, otherwise, False
        if len(positions) > 1:
            return True
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

    # def __enter__(self):
    #     self.connect_server()
    #     print("MetaTrader 5 is connected. ")
    #
    # def __exit__(self, *args):
    #     self.disconnect_server()
    #     print("MetaTrader 5 is disconnected. ")