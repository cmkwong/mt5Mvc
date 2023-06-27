import numpy as np
import pandas as pd
from models.myBacktest import pointsModel
from models.myUtils import mathsModel

class Base:

    def get_modified_coefficient_vector(self, coefficient_vector, long_mode, lot_times=1):
        """
        :param coefficient_vector: np.array, if empty array, it has no coefficient vector -> 1 or -1
        :param long_mode: Boolean, True = long spread, False = short spread
        :return: np.array
        """
        if long_mode:
            modified_coefficient_vector = np.append(-1 * coefficient_vector[1:], 1)  # buy real, sell predict
        else:
            modified_coefficient_vector = np.append(coefficient_vector[1:], -1)  # buy predict, sell real
        return modified_coefficient_vector.reshape(-1, ) * lot_times

    def get_coefficient_vector(self, input, target):
        """
        :param input: array, size = (total_len, )
        :param target: array, size = (total_len, )
        :return: coefficient
        """
        A = np.concatenate((np.ones((len(input), 1), dtype=float), input.reshape(len(input), -1)), axis=1)
        b = target.reshape(-1, 1)
        A_T_A = np.dot(np.transpose(A), A)
        A_T_b = np.dot(np.transpose(A), b)
        coefficient_vector = np.dot(np.linalg.inv(A_T_A), A_T_b)
        return coefficient_vector

    def get_predicted_arr(self, input, coefficient_vector):
        """
        Ax=b
        :param input: array, size = (total_len, feature_size)
        :param coefficient_vector: coefficient vector, size = (feature_size, )
        :return: predicted array
        """
        A = np.concatenate((np.ones((len(input), 1)), input.reshape(len(input), -1)), axis=1)
        b = np.dot(A, coefficient_vector.reshape(-1, 1)).reshape(-1, )
        return b

    def get_coin_data(self, inputs, coefficient_vector, mean_window, std_window):
        """
        :param inputs: accept the train and test prices in pd.dataframe format
        :param coefficient_vector:
        :return:
        """
        coin_data = pd.DataFrame(index=inputs.index)
        coin_data['real'] = inputs.iloc[:, -1]
        coin_data['predict'] = self.get_predicted_arr(inputs.iloc[:, :-1].values, coefficient_vector)
        spread = coin_data['real'] - coin_data['predict']
        coin_data['spread'] = spread
        coin_data['z_score'] = mathsModel.z_score_with_rolling_mean(spread.values, mean_window, std_window)
        return coin_data

    def get_strategy_id(self, train_options):
        id = 'coin'
        for key, value in train_options.items():
            id += str(value)
        long_id = id + 'long'
        short_id = id + 'short'
        return long_id, short_id

    def get_ret_earning(self, new_prices, old_prices, modify_exchg_q2d, points_dff_values_df, coefficient_vector, long_mode, lot_times=1):  # see note (45a)
        """
        :param new_prices: pd.DataFrame
        :param old_prices: pd.DataFrame
        :param modify_exchg_q2d: pd.Dataframe, that exchange the dollar into same deposit assert
        :param points_dff_values_df: points the change with respect to quote currency
        :param coefficient_vector: np.array
        :param long_mode: Boolean
        :param lot_times: lot times
        :return: pd.Series, pd.Series
        """
        modified_coefficient_vector = self.get_modified_coefficient_vector(coefficient_vector, long_mode, lot_times)

        # ret
        change = (new_prices - old_prices) / old_prices
        olds = np.sum(np.abs(modified_coefficient_vector))
        news = (np.abs(modified_coefficient_vector) + (change * modified_coefficient_vector)).sum(axis=1)
        ret = pd.Series(news / olds, index=new_prices.index, name="return")

        # earning
        weighted_pt_diff = points_dff_values_df.values * modified_coefficient_vector.reshape(-1, )
        # calculate the price in required deposit dollar
        earning = pd.Series(np.sum(modify_exchg_q2d.values * weighted_pt_diff, axis=1), index=modify_exchg_q2d.index, name="earning")  # see note 34b and 35 why shift(1)

        return ret, earning

    def get_value_of_ret(self, new_values, old_values, modified_coefficient_vector):
        # ret value
        changes = (new_values - old_values) / old_values
        olds = np.sum(np.abs(modified_coefficient_vector))
        news = (np.abs(modified_coefficient_vector) + (changes * modified_coefficient_vector)).sum()
        ret = news / olds
        return ret

    def get_value_of_earning(self, symbols, new_values, old_values, q2d_at, all_symbols_info, modified_coefficient_vector):
        """
        :param symbols: [str]
        :param new_values: np.array
        :param old_values: np.array
        :param q2d_at: np.array
        :param all_symbols_info: nametuple
        :param modified_coefficient_vector: np.array
        :return: float
        """
        if isinstance(symbols, str): symbols = [symbols]
        if isinstance(new_values, (float, int)): new_values = np.array([new_values])
        if isinstance(old_values, (float, int)): old_values = np.array([old_values])
        if isinstance(q2d_at, (float, int)): q2d_at = np.array([q2d_at])

        # earning value
        points_dff_values = pointsModel.get_points_dff_values_arr(symbols, new_values, old_values, all_symbols_info)
        weighted_pt_diff = points_dff_values * modified_coefficient_vector.reshape(-1, )
        # calculate the price in required deposit dollar
        earning = np.sum(q2d_at * weighted_pt_diff)
        return earning

    def get_value_of_ret_earning(self, symbols, new_values, old_values, q2d_at, all_symbols_info, lot_times, coefficient_vector, long_mode):
        """
        This is calculate the return and earning from raw value (instead of from dataframe)
        :param symbols: [str]
        :param new_values: np.array (Not dataframe)
        :param old_values: np.array (Not dataframe)
        :param q2d_at: np.array, values at brought the assert
        :param coefficient_vector: np.array
        :param all_symbols_info: nametuple
        :param long_mode: Boolean
        :return: float, float: ret, earning
        """
        if not isinstance(symbols, list):
            symbols = [symbols]
        if not isinstance(new_values, np.ndarray):
            new_values = np.array([new_values])
        if not isinstance(old_values, np.ndarray):
            old_values = np.array([old_values])
        if not isinstance(q2d_at, np.ndarray):
            q2d_at = np.array([q2d_at])

        modified_coefficient_vector = self.get_modified_coefficient_vector(coefficient_vector, long_mode, lot_times)

        # ret value
        ret = self.get_value_of_ret(new_values, old_values, modified_coefficient_vector)

        # earning value
        earning = self.get_value_of_earning(symbols, new_values, old_values, q2d_at, all_symbols_info, modified_coefficient_vector)

        return ret, earning