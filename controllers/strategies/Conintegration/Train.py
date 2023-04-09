from models.myUtils import strModel
from models.myBacktest import coinModel


class Train:
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsApiController
        self.plotController = mainController.plotController

    def simpleCheck(self, *, symbols: list, start: tuple, end: tuple, timeframe: str, outputPath: str):
        """
        :param symbols: ["AUDJPY", "USDCAD"]
        :param start: (2023, 2, 1, 0, 0)
        :param end: (2023, 2, 28, 23, 59)
        :param timeframe: 1H
        :return:
        """
        # get Prices
        Prices = self.mt5Controller.mt5PricesLoader.getPrices(symbols=symbols, start=start, end=end, timeframe=timeframe, count=0, ohlcvs='111100')
        # changes plotting
        cc = Prices.cc
        cc_coef = coinModel.get_coefficient_vector(cc.values[:, :-1], cc.values[:, -1])
        cc['residual'] = (cc * cc_coef.reshape(-1,)).sum(axis=1)
        ccFilename = "cc_" + "-".join(symbols) + f"_{timeframe}.png"
        self.plotController.plotSimpleLine(cc['residual'], 'cc residual', outputPath, ccFilename)

        # close price plotting
        close = Prices.close
        close_coef = coinModel.get_coefficient_vector(close.values[:, :-1], close.values[:, -1])
        close['residual'] = (close * close_coef.reshape(-1, )).sum(axis=1)
        closeFilename = "close_" + "-".join(symbols) + f"_{timeframe}.png"
        self.plotController.plotSimpleLine(close['residual'], 'close residual', outputPath, closeFilename)

        print()
        print("cc_coef: ", " + ".join(["({:.2f}){}".format(v, symbols[i]) for i, v in enumerate(cc_coef.reshape(-1,).tolist())]) + " = 0")
        print("close_coef: ", " + ".join("({:.2f}){}".format(v, symbols[i]) for i, v in enumerate(close_coef.reshape(-1,).tolist())) + " = 0")

