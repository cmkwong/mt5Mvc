import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import table
import os
import seaborn as sns
import numpy as np


# def testing(f):
#     def wrapper(*args):
#         print(args[0].url)
#         return f(*args)
#     print(f)
#     return wrapper
#     # for k, v in kwargs.items():
#     #     print(k, v)

class PlotController:
    def __init__(self, adjustTimeResolution=False, figsize=(10, 6), dpi=150, preview=False):
        self.locator = self.getLocator()
        self.formatter = self.getFormatter(self.locator)
        self.adjustTimeResolution = adjustTimeResolution
        self.figsize = figsize
        self.dpi = dpi
        # self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.preview = preview

    def getLocator(self):
        # matplotlib format setup
        # https://stackoverflow.com/questions/15165065/matplotlib-datetime-xlabel-issue
        locator = mdates.AutoDateLocator()
        locator.intervald[mdates.YEARLY] = [1, 2, 4, 5, 10]
        locator.intervald[mdates.MONTHLY] = [1, 2, 3, 4, 6]
        locator.intervald[mdates.DAILY] = [1, 2, 3, 4, 7, 14]
        locator.intervald[mdates.HOURLY] = [1, 2, 3, 6]
        locator.intervald[mdates.MINUTELY] = [1, 5, 10, 15, 30]
        locator.intervald[mdates.SECONDLY] = [1, 5, 10, 15, 30]
        return locator

    def getFormatter(self, locator):
        formatter = mdates.ConciseDateFormatter(locator, show_offset=False)  # formatter is disabled
        formatter.formats = [
            "%y",
            "%d %b %y",
            "%d",
            "%H:%M",
            "%H:%M",
            "%S.%f",
        ]
        formatter.zero_formats = [""] + formatter.formats[:-1]
        formatter.offset_formats = [
            "",
            "%Y",
            "%b %Y",
            "%d %b %Y",
            "%d %b %Y",
            "%d %b %Y %H:%M",
        ]
        return formatter

    # get the dataframe timestep in seconds
    def getTimestep(self, df):
        totalDiff_second = df.index.to_series().diff().dt.seconds.fillna(0).sum() / (len(df) - 1)
        return totalDiff_second

    # get required time resolution to display based on the start and end date range
    def getRequiredTimeResolution(self, df, adjust=True):
        # if adjust is False, then then the original width and daily sample

        # calculate the time difference between start and end
        totalDiff_second = df.index.to_series().diff().dt.seconds.fillna(0).sum()
        totalDiff_day = totalDiff_second / 3600 / 24

        # check if larger than 10 day, if so, then resample as 'D'
        if (totalDiff_day > 10.0) or not adjust:
            width = 0.35
            resampleFactor = 'D'
            minsDiff = 1440
        else:
            width = 0.02
            resampleFactor = '1H'
            minsDiff = 60
        return width, resampleFactor, minsDiff

    def plotHist(self, series, outPath, filename, bins=100, quantiles=(0.25, 0.5, 0.75)):
        """
        :param series: pd.Series
        :param outPath: str
        :param filename: str
        :param bins: int
        :param quantiles: set, eg: (0.25, 0.5, 0.75)
        :return:
        """
        # reset axis
        plt.clf()

        # add sub-plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.hist(series.values, bins=bins)

        # if plotting need to view on quantiles
        if quantiles:
            qvs = np.quantile(series.values, quantiles)
            for i in range(len(quantiles)):
                plt.axvline(qvs[i], color='k', linestyle='dashed', linewidth=1)
                plt.text(qvs[i] + (qvs[i] * 0.1), 0, f"{qvs[i]:.3g}", rotation=90, fontsize='x-small')

        imgFullPath = os.path.join(outPath, filename)
        fig.savefig(imgFullPath, bbox_inches="tight", transparent=True)
        plt.close('all')

    def plotBar(self, series, width, yLabel, color, edgecolor, outPath, filename, binUnit='', ylim=False, decimal=1):
        # reset axis
        plt.clf()

        # add sub-plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.bar(series.index, series, width=width, color=color, edgecolor=edgecolor)

        # rotate the x-axis
        plt.xticks(rotation=90)

        ax.xaxis.set_major_locator(self.locator)
        # ax.xaxis.set_major_formatter(self.formatter)
        ax.set_ylabel(yLabel)
        ax.set_facecolor("whitesmoke")

        max_height = series.max()
        min_height = series.min()

        # set the graph max height
        try:
            if ylim: ax.set_ylim((min_height * 0.9, max_height * 1.1))
        except ValueError:
            pass
        for p in ax.patches:
            value = p.get_height()
            valuePer = (value - min_height) / (max_height - min_height)
            if valuePer > 0.05:
                if valuePer > 0.90:
                    va = "top"
                    txtColor = "white"
                else:
                    va = "bottom"
                    txtColor = "black"
                ax.text(
                    p.get_x() + (p.get_width() * 0.6),
                    value,
                    s=" {0:.{1}f}".format(value, decimal) + binUnit,
                    ha="center",
                    va=va,
                    rotation=90,
                    color=txtColor,
                )
        fig.tight_layout()
        imgFullPath = os.path.join(outPath, filename)
        fig.savefig(imgFullPath, bbox_inches="tight", transparent=True)
        plt.close('all')
        return imgFullPath

    def plotMultiHistogramWithSum(self, df, yLabel, outPath, filename):
        # reset axis
        plt.clf()
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # define the bar width
        width = 0.3
        # calculate the width assignment table
        widthTable = [w * width for w in np.arange(len(df.columns))]  # [0, 0.2, 0.4, 0.6]
        widthTable = np.array(widthTable) - np.mean(widthTable)  # [-0.3, -0.1, 0.1, 0.3]
        x = np.arange(0, len(df.index) * 2, 2)  # make it wider when step is 2
        for i, colName in enumerate(df):
            ax.bar(x + widthTable[i], df[colName], width=width, label=colName)

        # calculate the sum and plot it
        sumValues = df.sum(axis=1)
        ax.plot(x, sumValues, label='Total')

        # reassign the x-ticks
        ax.set_xticks(x, df.index)
        plt.xticks(rotation=90)

        # set axis
        ax.xaxis.set_major_locator(self.locator)
        ax.set_ylabel(yLabel)
        ax.legend(loc='upper left')

        # set layout
        fig.tight_layout()
        plt.autoscale(enable=True, axis='x', tight=True)
        imgFullPath = os.path.join(outPath, filename)
        fig.savefig(imgFullPath, bbox_inches="tight", transparent=True)
        plt.close('all')
        return imgFullPath

    def plotSimpleLine(self, series, yLabel, outPath, filename):
        # reset axis
        # plt.cla()

        # add sub-plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(series.index, series)
        ax.xaxis.set_major_formatter(self.formatter)

        # rotate the x-axis
        plt.xticks(rotation=90)
        ax.xaxis.set_major_locator(self.locator)

        # set ylabel
        ax.set_ylabel(yLabel)

        fig.tight_layout()
        imgFullPath = os.path.join(outPath, filename)
        fig.savefig(imgFullPath, bbox_inches="tight", transparent=True)
        plt.close('all')
        return imgFullPath

    def plotMultiLine(self, df, yLabel, outPath, filename):
        # reset axis
        plt.clf()
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # plot graph
        for colName in df:
            ax.plot(df.index, df[colName], label=colName)

        plt.xticks(rotation=90)
        ax.xaxis.set_major_locator(self.locator)
        ax.set_ylabel(yLabel)
        ax.legend(loc='upper left')
        fig.tight_layout()
        imgFullPath = os.path.join(outPath, filename)
        fig.savefig(imgFullPath, bbox_inches="tight", transparent=True)
        plt.close('all')
        return imgFullPath

    # plot head-map
    def plotCorrHeatMap(self, df, outPath, filename):
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.tight_layout()
        imgFullPath = os.path.join(outPath, filename)
        plt.savefig(imgFullPath, bbox_inches="tight", transparent=True)
        plt.close('all')
        return imgFullPath

    # save df into img
    def df2Img(self, df, path, filename):
        plt.clf()
        fig, ax = plt.subplots(111, figsize=self.figsize, dpi=self.dpi, frame_on=False)  # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        table(ax, df, loc='center')  # where df is your data frame
        # save img
        fullPath = os.path.join(path, filename)
        fig.savefig(fullPath, bbox_inches='tight')
        plt.close('all')
        return fullPath

    # @testing
    def getGafImg(self, x_gasf, x_gadf, series, filename):
        """
        :param x: np.array
        :param filename: str
        """
        # reset axis
        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), dpi=self.dpi)

        index = series.reset_index().index
        # set limit
        ax1.set_xlim([index.min(), index.max()])
        # plot graph
        ax1.plot(index, series)
        # show image
        c = ax2.imshow(x_gasf, origin='lower', aspect='auto')
        ax2.text(5, 5, 'gasf')
        # fig.colorbar(c, ax=ax2, orientation="horizontal")
        c = ax3.imshow(x_gadf, origin='lower', aspect='auto')
        ax3.text(5, 5, 'gadf')
        # fig.colorbar(c, ax=ax3, orientation="horizontal")

        fig.tight_layout()
        fig.savefig(f'./docs/img/{filename}')
        plt.close('all')

    # def imgSave2(self, arr2D, series, filename):
    #     """
    #     :param x: np.array
    #     :param filename: str
    #     """
    #     # reset axis
    #     plt.clf()
    #     fig = plt.figure(figsize=self.figsize)
    #     gs = fig.add_gridspec(2, 1, height_ratios=(5, 1))
    #     fig.add
