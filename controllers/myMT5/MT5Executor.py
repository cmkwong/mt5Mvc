import config
import MetaTrader5 as mt5

"""
This is pure executor that will not store any status of trades
The status will be stored in NodeJS server, including:

self-defined:
    StrategyId

MetaTrader:
    positionId
    
"""


class MT5Executor:
    def __init__(self, all_symbols_info, get_historical_deals, get_position_volume_balance):
        self.all_symbols_info = all_symbols_info
        self._fn_get_historical_deals = get_historical_deals
        self._fn_get_position_volume_balance = get_position_volume_balance

    def request_format(self, symbol, operation, deviation, lot, sltp=(), pt_sltp=(), comment=''):
        """
        :param strategy_id: str, belong to specific strategy
        :param lots: [float]
        :param close_pos: Boolean, if it is for closing position, it will need to store the position id for reference
        :return: requests, [dict], a list of request
        """
        # type of filling
        tf = None
        if config.TypeFilling == 'fok':
            tf = mt5.ORDER_FILLING_FOK
        elif config.TypeFilling == 'ioc':
            tf = mt5.ORDER_FILLING_IOC
        elif config.TypeFilling == 'return':
            tf = mt5.ORDER_FILLING_RETURN

        # building request format
        if operation == 'long':
            action_type = mt5.ORDER_TYPE_BUY  # int = 0
            price = mt5.symbol_info_tick(symbol).ask
        elif operation == 'short':
            action_type = mt5.ORDER_TYPE_SELL  # int = 1
            price = mt5.symbol_info_tick(symbol).bid
            # lot = -lot
        else:
            raise Exception("The lot cannot be 0")  # if lot equal to 0, raise an Error
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': float(lot),
            'type': action_type,
            'price': price,
            'deviation': deviation,  # indeed, the deviation is useless when it is marketing order, note 73d
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": tf,
            "comment": comment
        }
        # transfer into sltp from pt if needed
        if pt_sltp:
            sltp = self.transfer_sltp_from_pt(symbol, price, pt_sltp, operation)
        # get the sltp
        if sltp and sltp[0] > 0:
            request['sl'] = sltp[0]
        if sltp and sltp[1] > 0:
            request['tp'] = sltp[1]

        return request

    def transfer_sltp_from_pt(self, symbol: str, price: float, pt_sltp: tuple, operation: str):
        """
        sltp change into price value if passed into point
        calculate the stop loss and take profit
        """
        sltp = [0, 0]
        # get factor +1 or -1
        SLTP_FACTOR = 1 if operation == 'long' else -1
        # get digit by symbol
        digit = self.all_symbols_info[symbol]['digits']  # set the digit
        if pt_sltp and pt_sltp[0] and pt_sltp[0] > 0:
            sltp[0] = price - SLTP_FACTOR * (pt_sltp[0] * (10 ** (-digit)))
        if pt_sltp and pt_sltp[1] and pt_sltp[1] > 0:
            sltp[1] = price + SLTP_FACTOR * (pt_sltp[1] * (10 ** (-digit)))
        return tuple(sltp)

    def close_request_format(self, *, position_id: int, percent: float = 1.0, comment: str = ''):
        """
        return close the position request format
        """
        # get the first deal which is open position
        openDeals = self._fn_get_historical_deals(position_id=position_id, datatype=dict)
        # get the first deal (open position)
        if openDeals:
            openDeal = openDeals[0]
        else:
            print(f"Cannot find position id: {position_id}")
            return False
        # get the information
        symbol = openDeal['symbol']
        oppositeType = 1 if openDeal['type'] == 0 else 0

        # check if volume is not exceed the balance
        volume_balance = self._fn_get_position_volume_balance(position_id)
        volume = openDeal['volume'] * percent
        volume = min(volume, volume_balance)

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'type': oppositeType,
            'price': mt5.symbol_info_tick(symbol).bid,
            'symbol': symbol,
            'volume': volume,
            'position': position_id,
            'comment': comment
        }
        return request

    def request_execute(self, request):
        """
        :param request: request
        :return: Boolean
        """
        if not request:
            return False

        # sending the request
        result = mt5.order_send(request)

        # get the basic info
        symbol = request['symbol']
        requestPrice = request['price']
        resultPrice = result.price
        typeLabel = 'long' if request['type'] == 0 else 'short'
        digit = mt5.symbol_info(request['symbol']).digits

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"order_send failed, symbol={symbol}, retcode={result.retcode}"
            if (result.retcode in config.MT5_ERROR_CODE.keys()):
                msg += f" - {config.MT5_ERROR_CODE[result.retcode]['description']}"
            print(msg)
            return False
        print(f"Action: {typeLabel}; by {symbol} {result.volume:.2f} lots at {result.price:.5f} ( ptDiff={((requestPrice - resultPrice) * 10 ** digit):.1f} ({requestPrice:.5f}("
              f"request.price) - {result.price:.5f}(result.price) ))")
        return result
