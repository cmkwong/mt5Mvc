from mt5Mvc.models.myUtils.paramModel import DatetimeTuple
from mt5Mvc.models.myUtils import timeModel, fileModel
from mt5Mvc.controllers.strategies.MovingAverage.Base import Base

class Backtest(Base):
    def __init__(self, mt5Controller, nodeJsApiController, plotController):
        self.mt5Controller = mt5Controller
        self.nodeJsApiController = nodeJsApiController
        self.plotController = plotController

    def getMaDistImg(self, curTime=None, *,
                     symbol: str = 'USDJPY',
                     timeframe: str = '15min',
                     start: DatetimeTuple = (2023, 5, 1, 0, 0),
                     end: DatetimeTuple = (2023, 6, 30, 23, 59),
                     fast: int = 14,
                     slow: int = 22,
                     operation: str = 'long'
                     ):

        # create folder
        if not curTime:
            curTime = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        distPath = fileModel.createDir(self.DistributionPath, curTime)

        # getting time string
        startStr = timeModel.getTimeS(start, '%Y-%m-%d %H-%M')
        endStr = timeModel.getTimeS(end, '%Y-%m-%d %H-%M')

        # getting the ma data
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol], start=start, end=end, timeframe=timeframe)

        MaDist = self.getMaDist(Prices, fast, slow)

        # output image
        for symbol, operations in MaDist.items():
            for op, dists in operations.items():
                # only need the operation specific from argument
                if op != operation: continue
                # create the axs
                axs = self.plotController.getAxes(len(dists), 1, (20, 70))
                for i, (distName, dist) in enumerate(dists.items()):
                    self.plotController.plotHist(axs[i], dist, distName, custTexts={'start': startStr, 'end': endStr, 'timeframe': timeframe, 'fast': fast, 'slow': slow, 'operation': op}, mean=True)
                    # distPath, f'{symbol}-{operation}-{startStr}-{endStr}-{distName}.jpg'
                self.plotController.saveImg(distPath, f'{symbol}-{op}-{fast}-{slow}-{timeframe}-{startStr}-{endStr}.jpg')
