from mt5Mvc.models.myUtils import paramModel, printModel, inputModel
import pandas as pd
from datetime import datetime, timedelta

class Handler_Deal:
    def __init__(self, nodeJsApiController, mt5Controller, stockPriceLoader, threadController, strategyController, plotController):
        self.nodeJsApiController = nodeJsApiController
        self.mt5Controller = mt5Controller
        self.stockPriceLoader = stockPriceLoader
        self.threadController = threadController
        self.strategyController = strategyController
        self.plotController = plotController

    def run(self, command):

        # close all deals
        if command == '-close':
            # ask if close all
            if inputModel.askConfirm("Close all the deals triggered by strategy? "):
                for id, strategy in self.strategyController:
                    # if succeed to close deal
                    if strategy.closeDeal(comment='Force to Close'):
                        print(f"Strategy id: {id} closed. ")
            else:
                paramFormat = {
                    "position_id": [0, int],
                    "percent": [1.0, float],
                    "comment": ["Manuel Close", str]
                }
                param = paramModel.ask_param(**paramFormat)
                request = self.mt5Controller.executor.close_request_format(**param)
                self.mt5Controller.executor.request_execute(request)

        # get the historical deals
        elif command == '-deals':
            deals = self.mt5Controller.get_historical_deals(lastDays=1)
            printModel.print_df(deals)

        # display the positions
        elif command == '-positions':
            positionsDf = self.mt5Controller.get_active_positions()
            nextTargetDf = self.nodeJsApiController.executeMySqlQuery('query_positions_next_target')
            # merge the positionsDf and nextTargetDf
            if not nextTargetDf.empty:
                positionsDf['ticket'] = positionsDf['ticket'].astype('str')
                positionsDf = pd.merge(positionsDf, nextTargetDf, left_on='ticket', right_on='position_id', how='left', right_index=False).fillna('')
                # positionsDf['position_id'] = positionsDf['position_id'].astype('Int64').astype('str')
            printModel.print_df(positionsDf)
            # account balance
            self.mt5Controller.print_account_balance()

        # check the deal performance from-to
        elif command == '-performance':
            now = datetime.now()
            dateFormat = "%Y-%m-%d %H:%M:%S"
            paramFormat = {
                'datefrom': ((now + timedelta(hours=-48)).strftime(dateFormat), str),
                'dateto': (now.strftime(dateFormat), str)
            }
            param = paramModel.ask_param(paramFormat)
            df = self.nodeJsApiController.executeMySqlQuery('query_position_performance', param)
            printModel.print_df(df)
            # account balance
            self.mt5Controller.print_account_balance()

        else:
            return True