
class ForexEnvController:
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsServerController

    def setUpState(self, state):
        pass

    def step(self, action):
        pass