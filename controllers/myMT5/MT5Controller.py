from controllers.myMT5.MT5PricesLoader import MT5PricesLoader
from controllers.myMT5.MT5Executor import MT5Executor
from controllers.myMT5.MT5SymbolController import MT5SymbolController
from controllers.myMT5.MT5TickController import MT5TickController
from controllers.myMT5.MT5TimeController import MT5TimeController
from models.myUtils import dfModel, printModel

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import config


class MT5Controller:
    def __init__(self, nodeJsApiController):
        self.connect_server()
        self.symbolController = MT5SymbolController()
        self.tickController = MT5TickController()
        self.timeController = MT5TimeController()
        self.pricesLoader = MT5PricesLoader(self.timeController, self.symbolController, nodeJsApiController)  # loading the loader
        self.executor = MT5Executor(self.pricesLoader.all_symbols_info, self.get_historical_deals)  # execute the request (buy/sell)

    def print_terminal_info(self):
        # request connection status and parameters
        print(mt5.terminal_info())
        # request account info
        print(mt5.account_info())
        # get loader on MetaTrader 5 version
        print(mt5.version())

    def get_active_positions(self):
        """
        :return: print all of the active order situation
        """
        cols = ['ticket', 'time', 'symbol', 'volume', 'type', 'profit', 'price_open', 'price_current']
        postions = mt5.positions_get()
        datas = {}
        for i, position in enumerate(postions):
            datas[i + 1] = []
            for col in cols:
                v = getattr(position, col)
                if col == 'time':
                    v = datetime.fromtimestamp(v) + timedelta(hours=-8)
                datas[i + 1].append(v)
        positionsDf = pd.DataFrame.from_dict(datas, orient='index', columns=cols)
        return positionsDf

    def get_historical_deals(self, *, position_id: int = None, lastDays: int = 365, datatype=pd.DataFrame):
        """
        :param position_id: if specify, then ignore the date range
        :param lastDays: 1625 = 5 years
        :param datatype: pd.DataFrame / dict
        :return pd.Dataframe of deals
        """
        cols = ['ticket', 'time', 'order', 'type', 'entry', 'magic', 'position_id', 'reason', 'volume', 'price', 'commission', 'swap', 'profit', 'fee', 'symbol', 'comment', 'external_id']
        if position_id:
            deals = mt5.history_deals_get(position=position_id)
            if not deals:
                return None
        else:
            currentDate = datetime.today() + timedelta(hours=8)  # time object
            # currentDate = pytz.timezone('Hongkong').localize(datetime.today())
            fromDate = currentDate - timedelta(days=lastDays)
            deals = mt5.history_deals_get(fromDate, currentDate)
        datas = {}
        for i, deal in enumerate(deals):
            # time = datetime.fromtimestamp(deal.time)
            row = []
            for col in cols:
                row.append(getattr(deal, col))
            # append the
            datas[i] = row
        historicalDeals = pd.DataFrame.from_dict(datas, orient='index', columns=cols)
        # sort by date
        historicalDeals.sort_values(by=['time'], inplace=True)
        # transfer seconds into time format
        raw_datetime = historicalDeals['time'].apply(datetime.fromtimestamp)
        # resume back to UTC time
        raw_datetime = raw_datetime.apply(lambda t: t + timedelta(hours=-8))
        # set into date and time
        historicalDeals['date'] = raw_datetime.apply(lambda x: x.date())
        historicalDeals['time'] = raw_datetime.apply(lambda x: x.time())
        if datatype == dict:
            historicalDeals = historicalDeals.to_dict(orient='records')
        return historicalDeals

    def get_position_performace(self, position_id):
        # get required deal in last 1 year
        deals = self.get_historical_deals(position_id=position_id)
        if not deals:
            return None
        # sum all of the profit with same position id
        profits = deals['profit'].sum()
        swap = deals['swap'].sum()
        commission = deals['commission'].sum()
        duration = self.get_position_duration(position_id)
        return profits, swap, commission, duration

    def get_position_earn(self, position_id):
        # get required deal in last 1 year
        deals = self.get_historical_deals(position_id=position_id)
        if not deals:
            return None
        # sum all of the profit with same position id
        profits = deals.groupby('position_id')['profit'].sum()
        # if position_id not in profits.index:
        #     print(f"The required Position Id: {position_id} is not existed. ")
        return profits[position_id]

    def get_position_duration(self, position_id):
        """
        get the duration in time format (00: 00: 00)
        """
        # get all the positions with same position id
        orders = mt5.history_orders_get(position=position_id)
        durations = []
        for order in orders:
            # append the duration
            durations.append(order.time_done)
        # calculate the duration taken
        seconds = max(durations) - min(durations)
        duration = timedelta(seconds=seconds)
        return duration

    def get_position_cost(self, position_id):
        deals = self.get_historical_deals(position_id=position_id)
        swap = deals.groupby('position_id')['swap'].sum()
        commission = deals.groupby('position_id')['commission'].sum()
        return swap, commission

    def get_historical_order(self, *, position_id: int = None, lastDays: int = 10):
        """
        Rare to use
        """
        if position_id:
            orders = mt5.history_orders_get(position=position_id)
            if not orders:
                return None
        else:
            currentDate = datetime.today()  # time object
            fromDate = currentDate - timedelta(days=lastDays)
            orders = mt5.history_orders_get(fromDate, currentDate)
        return orders

    def check_order_closed(self, position_id):
        """
        Check if order finished or not
        :param position_id: ticket ID, in metatrader position ID is same as ticket ID
        :return: Boolean
        """
        # get all the positions with same position id
        positions = mt5.history_deals_get(position=position_id)
        # order finished return True, otherwise, False
        volumeBalance = 0.0
        for position in positions:
            factor = 1 if position.type == 0 else -1
            volumeBalance += factor * position.volume
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

