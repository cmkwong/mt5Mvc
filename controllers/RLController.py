from models.AI.ForexState import ForexState

class RLController:
    def __init__(self, mainController):
        # assign controller
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsApiController = mainController.nodeJsApiController

        self.env = self.defineEnv()
        self.agent = self.defineAgent()

    def defineEnv(self):
        ForexState()

    def defineAgent(self):
        pass