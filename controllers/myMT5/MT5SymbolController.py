import MetaTrader5 as mt5
import collections

import config

class MT5SymbolController:
    def get_symbol_total(self):
        """
        :return: int: number of symbols
        """
        num_symbols = mt5.symbols_total()
        if num_symbols > 0:
            print("Total symbols: ", num_symbols)
        else:
            print("Symbols not found.")
        return num_symbols

    def get_symbols(self, group=None):
        """
        :param group: https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolsget_py, refer to this website for usage of group
        :return: tuple(symbolInfo), there are several property
        """
        if group:
            symbols = mt5.symbols_get(group)
        else:
            symbols = mt5.symbols_get()
        return symbols

    def get_all_symbols_info(self):
        """
        :return: dict[symbol] = collections.nametuple
        """
        symbols_info = {}
        symbols = mt5.symbols_get()
        for symbol in symbols:
            symbol_name = symbol.name
            # symbols_info[symbol_name] = collections.namedtuple("info", ['digits', 'base', 'quote', 'swap_long', 'swap_short', 'pt_value'])
            symbols_info[symbol_name] = {}
            symbols_info[symbol_name]['digits'] = symbol.digits
            symbols_info[symbol_name]['base'] = symbol.currency_base
            symbols_info[symbol_name]['quote'] = symbol.currency_profit
            symbols_info[symbol_name]['swap_long'] = symbol.swap_long
            symbols_info[symbol_name]['swap_short'] = symbol.swap_short
            if symbol_name[3:] == 'JPY':
                symbols_info[symbol_name]['pt_value'] = 100  # 100 dollar for quote per each point    (See note Stock Market - Knowledge - note 3)
            else:
                symbols_info[symbol_name]['pt_value'] = 1  # 1 dollar for quote per each point  (See note Stock Market - Knowledge - note 3)
        return symbols_info

    def get_exchange_symbols(self, symbols, exchg_type='q2d'):
        """
        Find all the currency pair related to and required currency and deposit symbol
        :param symbols: [str] : ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]
        :param exchg_type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
        :return: [str], get required exchange symbol in list: ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
        """
        all_symbols_info = self.get_all_symbols_info()
        symbol_names = list(all_symbols_info.keys())
        exchange_symbols = []
        target_symbol = None
        for symbol in symbols:
            if exchg_type == 'b2d':
                target_symbol = symbol[:3]
            elif exchg_type == 'q2d':
                target_symbol = symbol[3:]
            if target_symbol != config.DepositCurrency:  # if the symbol not relative to required deposit currency
                test_symbol_1 = target_symbol + config.DepositCurrency
                test_symbol_2 = config.DepositCurrency + target_symbol
                if test_symbol_1 in symbol_names:
                    exchange_symbols.append(test_symbol_1)
                    continue
                elif test_symbol_2 in symbol_names:
                    exchange_symbols.append(test_symbol_2)
                    continue
                else:  # if not found the relative pair with respect to deposit currency, raise the error
                    raise Exception("{} has no relative currency with respect to deposit {}.".format(target_symbol, config.DepositCurrency))
            else:  # if the symbol already relative to deposit currency
                exchange_symbols.append(symbol)
        return exchange_symbols