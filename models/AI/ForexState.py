from models.myBacktest import pointsModel
from models.myBacktest import techModel
from models.myUtils import dicModel
import config

import numpy as np
import pandas as pd
import random

class ForexState:
    def __init__(self, Prices, tech_params,
                 long_mode, all_symbols_info, reset_on_close):
        self._init_action_space()
        self.Prices = Prices
        self.symbol = Prices.symbols[0]
        self.tech_params = tech_params              # pd.DataFrame
        self.closePrice = Prices.close              # close price (pd.DataFrame)
        self.quote_exchg = Prices.quote_exchg       # quote to deposit (pd.DataFrame)
        self.long_mode = long_mode                  # Boolean
        self.all_symbols_info = all_symbols_info    # dict
        self.reset_on_close = reset_on_close        # Boolean
        self.time_cost_pt = config.TimeCostPt       # float, eg: 0.05
        self.commission_pt = config.CommisionPt     # float, eg: 8
        self.spread_pt = config.SpreadPt            # float, eg: 15
        self.deal_step = 0.0  # step counter from buy to sell (buy date = step 1, if sell date = 4, time cost = 3)
        # should be shift 1 forward, because it takes action on next-day of open-price (pd.DataFrame)
        self.dependent_datas = pd.concat([self._get_tech_df(Prices), Prices.open, Prices.high, Prices.low, Prices.close], axis=1, join='outer').fillna(0)

    def _get_tech_df(self, Prices):
        tech_df = pd.DataFrame()
        for tech_name in self.tech_params.keys():
            data = techModel.get_tech_datas(Prices, self.tech_params[tech_name], tech_name)
            tech_df = dicModel.append_dict_df(data, tech_df, join='outer', filled=0)
        return tech_df

    def _init_action_space(self):
        self.actions = {}
        self.actions['skip'] = 0
        self.actions['open'] = 1
        self.actions['close'] = 2
        self.action_space = list(self.actions.values())
        self.action_space_size = len(self.action_space)

    def reset(self, new_offset):
        # set offset if equal / larger than zero
        if new_offset >= 0:
            self._offset = new_offset
        # random index
        else:
            random_offset = np.random.randint(len(self.Prices.open) - 10)
            self._offset = random_offset
        self.have_position = False

    def cal_profit(self, curr_action_price, open_action_price, q2d_at):
        if self.long_mode:
            modified_coefficient_vector = 1
        else:
            modified_coefficient_vector = -1
        points_dff_values = pointsModel.get_points_dff_values_arr(self.symbol, curr_action_price, open_action_price, self.all_symbols_info)
        return np.sum(q2d_at * points_dff_values) * modified_coefficient_vector
        # return returnModel.get_value_of_earning(self.symbol, curr_action_price, open_action_price, q2d_at, self.all_symbols_info, modified_coefficient_vector)

    def encode(self):
        """
        :return: state
        """
        res = []
        earning = 0.0
        res.extend(list(self.dependent_datas.iloc[self._offset, :].values))
        if self.have_position:
            earning = self.cal_profit(self.closePrice.iloc[self._offset, :].values, self._prev_action_price, self.quote_exchg.iloc[self._offset, :].values)
        res.extend([earning, float(self.have_position)])  # earning, have_position (True = 1.0, False = 0.0)
        return np.array(res, dtype=np.float32)

    def step(self, action: int):
        """
        Calculate the rewards and check if the env is done
        :param action: long/short * Open/Close/hold position: 6 actions
        :return: reward, done
        """
        done = False
        reward = 0.0  # in deposit USD
        curr_action_price = self.closePrice.iloc[self._offset].values[0]
        q2d_at = self.quote_exchg.iloc[self._offset].values[0]

        if action == self.actions['open'] and not self.have_position:
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.spread_pt, q2d_at, self.all_symbols_info)  # spread cost
            self.openPos_price = curr_action_price
            self.have_position = True

        elif action == self.actions['close'] and self.have_position:
            reward += self.cal_profit(curr_action_price, self._prev_action_price, q2d_at)  # calculate the profit
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.time_cost_pt, q2d_at, self.all_symbols_info)  # time cost
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.spread_pt, q2d_at, self.all_symbols_info)  # spread cost
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.commission_pt, q2d_at, self.all_symbols_info)  # commission cost
            self.have_position = False
            if self.reset_on_close:
                done = True

        elif action == self.actions['skip'] and self.have_position:
            reward += self.cal_profit(curr_action_price, self._prev_action_price, q2d_at)
            reward -= pointsModel.get_point_to_deposit(self.symbol, self.time_cost_pt, q2d_at, self.all_symbols_info)  # time cost
            self.deal_step += 1

        # update status
        self._prev_action_price = curr_action_price
        self._offset += 1
        if self._offset >= len(self.closePrice) - 1:
            done = True

        return reward, done

# Prices, symbol, tech_params,
# time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close
class AttnForexState(ForexState):
    def __init__(self, seqLen, Prices, symbol, tech_params, time_cost_pt, commission_pt, spread_pt, long_mode, all_symbols_info, reset_on_close):
        super(AttnForexState, self).__init__(Prices, tech_params, long_mode, all_symbols_info, reset_on_close)
        self.seqLen = seqLen

    def reset(self, new_offset):
        # set offset if equal / larger than zero
        if new_offset >= 0:
            self._offset = new_offset
        # random index
        else:
            random_offset = random.randint(0, len(self.Prices.open) - self.seqLen)
            self._offset = random_offset
        self.have_position = False

    def encode(self):
        """
        :return: state
        """
        state = {}
        earning = 0.0
        if self.have_position:
            earning = self.cal_profit(self.closePrice.iloc[self._offset, :].values, self._prev_action_price, self.quote_exchg.iloc[self._offset, :].values)
        state['encoderInput'] = self.dependent_datas.iloc[self._offset:self._offset + self.seqLen, :].values  # getting seqLen * 2 len of Data
        state['status'] = np.array([earning, float(self.have_position)])  # earning, have_position (True = 1.0, False = 0.0)
        return state