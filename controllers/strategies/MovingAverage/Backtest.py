from models.myUtils.paramModel import SymbolList, DatetimeTuple
from models.myUtils import timeModel, fileModel
from controllers.strategies.MovingAverage.Base import Base
import config

import pandas as pd
import os

class Backtest(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.plotController = mainController.plotController

    def getMaDistImg(self, curTime = None, *,
                     symbols: list = config.DefaultSymbols,
                     timeframe: str = '15min',
                     start: DatetimeTuple = (2023, 5, 1, 0, 0),
                     end: DatetimeTuple = (2023, 6, 30, 23, 59),
                     fast_param: int = 14,
                     slow_param: int = 22):

        # create folder
        if not curTime:
            curTime = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        distPath = fileModel.createDir(self.DistributionPath, curTime)

        # getting time string
        startStr = timeModel.getTimeS(start, '%Y%m%d%H%M')
        endStr = timeModel.getTimeS(end, '%Y%m%d%H%M')

        # getting the ma data
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, start=start, end=end, timeframe=timeframe)
        MaData = self.getMaData(Prices, fast_param, slow_param)

        Distributions = self.getMaDist(MaData)

        # output image
        for symbol, operations in Distributions.items():
            for operation, dists in operations.items():
                # create the axs
                axs = self.plotController.getAxes(8, 1, (20, 60))
                for i, (distName, dist) in enumerate(dists.items()):
                    self.plotController.plotHist(axs[i], dist)
                    # distPath, f'{symbol}-{operation}-{startStr}-{endStr}-{distName}.jpg'
                self.plotController.saveImg(distPath, f'{symbol}-{operation}.jpg')

    def getMaDistImgs(self, *, versionNum: str = '2023-08-12 094616'):
        # get the cur time
        curTime = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        # read the file list in folder
        folderPath = os.path.join(self.SummaryPath, versionNum)
        summaryFiles = fileModel.getFileList(folderPath)
        # loop for each summary
        for summaryFile in summaryFiles:
            summaryDf = pd.read_excel(os.path.join(folderPath, summaryFile))
            requiredParams = summaryDf[summaryDf['reliable'] == 1]
            for i, requiredParam in requiredParams.iterrows():
                symbol = requiredParam['symbol']
                timeframe = requiredParam['timeframe']
                start = timeModel.getTimeT(requiredParam['start'], '%Y-%m-%d %H:%M')
                end = timeModel.getTimeT(requiredParam['end'], '%Y-%m-%d %H:%M')
                fast = requiredParam['fast']
                slow = requiredParam['slow']
                self.getMaDistImg(curTime=curTime, symbols=[symbol], timeframe=timeframe, start=start, end=end, fast_param=fast, slow_param=slow)
