from mt5Mvc.models.myUtils.paramModel import DatetimeTuple
from mt5Mvc.models.myUtils import timeModel, fileModel
from mt5Mvc.controllers.strategies.MovingAverage.Base import Base
from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from mt5Mvc.controllers.PlotController import PlotController

class Backtest(Base):
    def __init__(self):
        self.mt5Controller = MT5Controller()
        self.nodeJsApiController = NodeJsApiController()
        self.plotController = PlotController()

    # output MA distribution image
    def getMaDistImg(self, Prices, fast, slow, operation):

        # create the disPath and its dir
        curTime = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        distPath = fileModel.createDir(self.DistributionPath, curTime)

        # string of start and end datetime
        periodStart = timeModel.getTimeS(Prices.start_index, '%Y-%m-%d %H%M')
        periodEnd = timeModel.getTimeS(Prices.end_index, '%Y-%m-%d %H%M')

        # get the MA distribution
        MaDist = self.getMaDist(Prices, fast, slow)

        # output image
        for symbol, operations in MaDist.items():
            for op, dists in operations.items():
                # only need the operation specific from argument
                if op != operation: continue
                # create the axs
                axs = self.plotController.getAxes(len(dists), 1, (25, 87.5))
                for i, (distName, dist) in enumerate(dists.items()):
                    self.plotController.plotHist(axs[i], dist, distName, metrics={'start': periodStart, 'end': periodEnd, 'timeframe': Prices.timeframe, 'fast': fast, 'slow': slow, 'operation': op}, mean=True)
                    # distPath, f'{symbol}-{operation}-{startStr}-{endStr}-{distName}.jpg'
                self.plotController.saveImg(distPath, f'{symbol}-{op}-{fast}-{slow}-{Prices.timeframe}-{periodStart}-{periodEnd}.jpg')

    def getForexMaDistImg(self,
                          *,
                          symbol: str = 'USDJPY',
                          timeframe: str = '15min',
                          start: DatetimeTuple = (2023, 5, 1, 0, 0),
                          end: DatetimeTuple = (2023, 6, 30, 23, 59),
                          fast: int = 14,
                          slow: int = 22,
                          operation: str = 'long'
                          ):

        # getting the ma data
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=[symbol], start=start, end=end, timeframe=timeframe)

        # output image
        self.getMaDistImg(Prices, fast, slow, operation)
