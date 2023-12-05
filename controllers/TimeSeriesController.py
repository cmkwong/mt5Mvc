import pandas as pd
import numpy as np
import statsmodels.api as sm
from pyts.image import GramianAngularField

# from statsmodels.graphics import tsaplots
# from pandas import read_csv
# from matplotlib import pyplot
# from statsmodels.graphics.tsaplots import plot_acf
# import pyts

class TimeSeriesController:
    """
    About analysis the time series data
    """
    def __init__(self):
        pass

    def decompose_timeData(self, timeData: pd.Series):
        """
        https://www.kaggle.com/code/bextuychiev/advanced-time-series-analysis-decomposition/notebook#Advanced-Time-Series-Analysis-in-Python:-Seasonality-and-Trend-Analysis-(Decomposition),-Autocorrelation
        :param timeData: pd.Series
        decomposition time-series data into seasonal, trend and residual
        """
        decomposition = sm.tsa.seasonal_decompose(timeData, period=int(len(timeData) / 2))
        return decomposition.seasonal, decomposition.trend, decomposition.resid

    def get_corelaDf(self, timeDatas: pd.DataFrame, rowvar=False, bias=False):
        """
        :param timeDatas: normally it is change
        :param rowvar:
        :param bias:
        :return:
        """
        changes_arr = timeDatas.values
        cor_matrix = np.corrcoef(changes_arr, rowvar=rowvar, bias=bias)
        symbols = list(timeDatas.columns)
        corelaDf = pd.DataFrame(cor_matrix, index=symbols, columns=symbols)
        return corelaDf

# symbol = 'EURUSD'
# params = {'symbols': [symbol], 'start': (2018, 1, 1, 0 ,0), 'end': (2023, 5, 31,23,59), 'timeframe': '1H', 'count': 0, 'ohlcvs': '111111'}
#
# plotController = PlotController(preview=True)
# mainController = MainController()
#
# Prices = mainController.mt5Controller.mt5PricesLoader.getPrices(**params)
# seasonal, trend, resid = decomposeTimeSeries(Prices.close[symbol])
#
# plotController.plotSimpleLine(seasonal, 'value', '../../docs/img', f"{timeModel.get_current_time_string()}_seasonPlot.png")
# plotController.plotSimpleLine(trend, 'value', '../../docs/img', f"{timeModel.get_current_time_string()}_trendPlot.png")
# plotController.plotSimpleLine(resid, 'value', '../../docs/img', f"{timeModel.get_current_time_string()}_residualPlot.png")
# plot_acf(Prices.close[symbol])
# pyplot.show()
#
# # plotController.plotSimpleLine(Prices.close[symbol].autocorr(120), 'value', '../../docs/img', f"{timeModel.get_current_time_string()}_acf.png")
#
#
# print()
