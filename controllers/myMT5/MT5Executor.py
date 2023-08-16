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
    # def __init__(self):
    #     pass

    def request_format(self, symbol, operation, sl, tp, deviation, lot):
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
            'sl': sl,
            'tp': tp,
            'deviation': deviation,  # indeed, the deviation is useless when it is marketing order, note 73d
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": tf,
        }
        return request

    def close_request_format(self, openResult, percent=1.0):
        """
        return close the position request format
        """
        symbol = openResult.request.symbol
        positionId = openResult.order
        oppositeType = 1 if openResult.request.type == 0 else 0
        volume = openResult.volume * percent
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'type': oppositeType,
            'price': mt5.symbol_info_tick(symbol).bid,
            'symbol': symbol,
            'volume': volume,
            'position': positionId,
        }
        return request

    def request_execute(self, request):
        """
        :param request: request
        :return: Boolean
        """
        # sending the request
        result = mt5.order_send(request)

        # get the basic info
        symbol = request['symbol']
        requestPrice = request['price']
        resultPrice = result.price
        typeLabel = 'long' if request['type'] == 0 else 'short'
        digit = mt5.symbol_info(request['symbol']).digits

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("order_send failed, symbol={}, retcode={}".format(symbol, result.retcode))
            return False
        print('--------------------------------')
        print(f"Action: {typeLabel}; by {symbol} {result.volume:.2f} lots at {result.price:.5f} ( ptDiff={((requestPrice - resultPrice) * 10 ** digit):.1f} ({requestPrice:.5f}("
              f"request.price) - {result.price:.5f}(result.price) ))")
        print('--------------------------------')
        return result
