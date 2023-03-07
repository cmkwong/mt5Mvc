
class Live:
    def __init__(self, mt5Controller, nodeJsServerController, tg=None, *, symbol, auto=False):
        self.mt5Controller = mt5Controller
        self.nodeJsServerController = nodeJsServerController
        self.tg = tg
