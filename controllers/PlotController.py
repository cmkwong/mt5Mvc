import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np

class PlotController:
    def __init__(self, adjustTimeResolution=False, figsize=(10, 6), dpi=150):
        self.locator = self.getLocator()
        self.formatter = self.getFormatter(self.locator)
        self.adjustTimeResolution = adjustTimeResolution
        self.plt = plt
        self.fig = plt.figure(figsize=figsize, dpi=dpi)

    def getLocator(self):
        # matplotlib format setup
        locator = mdates.AutoDateLocator()
        locator.intervald[mdates.DAILY] = [1]
        locator.intervald[mdates.HOURLY] = [1, 2, 3, 6]
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

    def plotHistorgram(self, series, width, yLabel, color, edgecolor, outPath, filename, binUnit='', ylim=False, decimal=1):
        # reset axis
        self.plt.delaxes()

        # add sub-plot
        ax = self.fig.add_subplot()
        ax.bar(series.index, series, width=width, color=color, edgecolor=edgecolor)

        # rotate the x-axis
        self.plt.xticks(rotation=90)

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
        self.fig.tight_layout()
        image_name = os.path.join(outPath, filename)
        self.fig.savefig(image_name, bbox_inches="tight", transparent=True)

    def plotMultiHistogramWithSum(self, df, yLabel, outPath, filename):
        # reset axis
        self.plt.delaxes()
        ax = self.fig.add_subplot()

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
        self.plt.xticks(rotation=90)

        # set axis
        ax.xaxis.set_major_locator(self.locator)
        ax.set_ylabel(yLabel)
        ax.legend(loc='upper left')

        # set layout
        self.fig.tight_layout()
        self.plt.autoscale(enable=True, axis='x', tight=True)
        image_name = os.path.join(outPath, filename)
        self.fig.savefig(image_name, bbox_inches="tight", transparent=True)

    def plotSimpleLine(self, series, yLabel, outPath, filename):
        # reset axis
        self.plt.delaxes()

        # add sub-plot
        ax = self.fig.add_subplot()
        ax.plot(series.index, series)

        # rotate the x-axis
        self.plt.xticks(rotation=90)
        ax.xaxis.set_major_locator(self.locator)

        # set ylabel
        ax.set_ylabel(yLabel)

        self.fig.tight_layout()
        image_name = os.path.join(outPath, filename)
        self.fig.savefig(image_name, bbox_inches="tight", transparent=True)

    def plotMultiLine(self, df, yLabel, outPath, filename):
        # reset axis
        self.plt.delaxes()
        ax = self.fig.add_subplot()

        # plot graph
        for colName in df:
            ax.plot(df.index, df[colName], label=colName)

        self.plt.xticks(rotation=90)
        ax.xaxis.set_major_locator(self.locator)
        ax.set_ylabel(yLabel)
        ax.legend(loc='upper left')
        self.fig.tight_layout()
        image_name = os.path.join(outPath, filename)
        self.fig.savefig(image_name, bbox_inches="tight", transparent=True)