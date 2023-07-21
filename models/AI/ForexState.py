from models.myBacktest import pointsModel
from models.myBacktest import techModel
from models.myUtils import dicModel
import config

import numpy as np
import pandas as pd
import random


class ForexState:
    def __init__(self, Prices, tech_params, long_mode, resetOnDone):
        self._setup_action_space()
        self.Prices = Prices
        self.symbol = Prices.symbols[0]
        self.tech_params = tech_params  # pd.DataFrame
        self.long_mode = long_mode  # Boolean
        self.all_symbols_info = Prices.all_symbols_info  # dict
        self.resetOnDone = resetOnDone  # Boolean
        self.time_cost_pt = config.TIME_COST_POINT  # float, eg: 0.05
        self.commission_pt = config.COMMISSION_POINT  # float, eg: 8
        self.spread_pt = config.SPREAD_POINT  # float, eg: 15
        self.deal_step = 0.0  # step counter from buy to sell (buy date = step 1, if sell date = 4, time cost = 3)
        # should be shift 1 forward, because it takes action on next-day of open-price (pd.DataFrame)
        self.dependent_datas = pd.concat([self._get_tech_df(Prices), Prices.open, Prices.high, Prices.low, Prices.close], axis=1, join='outer').fillna(0)

    def _setup_action_space(self):
        self.actions = {}
        self.actions['skip'] = 0
        self.actions['open'] = 1
        self.actions['close'] = 2
        self.action_space = list(self.actions.values())
        self.action_space_size = len(self.action_space)

    def _get_tech_df(self, Prices):
        tech_df = pd.DataFrame()
        for tech_name in self.tech_params.keys():
            data = techModel.get_tech_datas(Prices, self.tech_params[tech_name], tech_name)
            tech_df = dicModel.append_dict_df(data, tech_df, join='outer', filled=0)
        return tech_df

    def reset(self, new_offset=None):
        # not random index
        if not new_offset:
            self._offset = new_offset
        else:
            # random index
            self._offset = np.random.randint(len(self.Prices.open) - 10)
        self.have_position = False

    def calProfit(self, offset_s, offset_e):
        if self.long_mode:
            modified_coefficient_vector = 1
        else:
            modified_coefficient_vector = -1
        values = self.Prices.getValueDiff(offset_s, offset_e).values
        # points_dff_values = pointsModel.get_points_dff_values_arr(self.symbol, curr_action_price, open_action_price, self.all_symbols_info)
        return np.sum(values) * modified_coefficient_vector

    def encode(self):
        """
        :return: state
        """
        res = []
        earning = 0.0
        res.extend(list(self.dependent_datas.iloc[self._offset, :].values))
        if self.have_position:
            earning = self.calProfit(self._open_offset, self._offset)[self.symbol]
            # earning = self.calProfit(self.Prices.close.iloc[self._offset, :].values, self._prev_action_price, self.Prices.quote_exchg.iloc[self._offset, :].values)
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
        curr_action_price = self.Prices.close.iloc[self._offset].values[0]
        q2d_at = self.Prices.quote_exchg.iloc[self._offset].values[0]

        if action == self.actions['open'] and not self.have_position:
            reward -= self.Prices.getPointValue(self.spread_pt, self._offset)[self.symbol]  # spread cost
            # self.openPos_price = curr_action_price
            self._open_offset = self._offset
            self.have_position = True

        elif action == self.actions['close'] and self.have_position:
            reward += self.calProfit(self._offset - 1, self._offset) # calculate the profit
            # reward += self.calProfit(curr_action_price, self._prev_action_price, q2d_at)  # calculate the profit
            reward -= self.Prices.getPointValue(self.time_cost_pt, self._offset)[self.symbol]  # time cost
            reward -= self.Prices.getPointValue(self.spread_pt, self._offset)[self.symbol]  # spread cost
            reward -= self.Prices.getPointValue(self.commission_pt, self._offset)[self.symbol]  # commission cost
            self.have_position = False
            if self.resetOnDone:
                done = True

        elif action == self.actions['skip'] and self.have_position:
            reward += self.calProfit(self._offset - 1, self._offset)
            # reward += self.calProfit(curr_action_price, self._prev_action_price, q2d_at)
            reward -= self.Prices.getPointValue(self.time_cost_pt, self._offset)[self.symbol]  # time cost
            self.deal_step += 1

        # update status
        # self._prev_action_price = curr_action_price
        self._offset += 1
        if self._offset >= len(self.Prices.close) - 1:
            done = True

        return reward, done


# attention network
class AttnForexState(ForexState):
    def __init__(self, Prices, tech_params, long_mode, resetOnDone, seqLen):
        super(AttnForexState, self).__init__(Prices, tech_params, long_mode, resetOnDone)
        self.seqLen = seqLen

    def reset(self, new_offset=None):
        # not random index
        if not new_offset:
            self._offset = new_offset
        # random index
        else:
            self._offset = random.randint(0, len(self.Prices.open) - self.seqLen)
        self.have_position = False

    def encode(self):
        """
        :return: state
        """
        state = {}
        earning = 0.0
        if self.have_position:
            earning = self.calProfit(self._offset - 1, self._offset)
            # earning = self.calProfit(self.Prices.close.iloc[self._offset, :].values, self._prev_action_price, self.Prices.quote_exchg.iloc[self._offset, :].values)
        state['encoderInput'] = self.dependent_datas.iloc[self._offset:self._offset + self.seqLen, :].values  # getting seqLen * 2 len of Data
        state['status'] = np.array([earning, float(self.have_position)])  # earning, have_position (True = 1.0, False = 0.0)
        return state
