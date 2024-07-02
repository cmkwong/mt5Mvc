
class Handler_Control:
    def __init__(self, nodeJsApiController, mt5Controller):
        self.nodeJsApiController = nodeJsApiController
        self.mt5Controller = mt5Controller

    def run(self, command):
        # switch the nodeJS server env: prod / dev
        if command == '-prod' or command == '-dev':
            self.nodeJsApiController.switchEnv(command[1:])

        # switch the price loader source from mt5 / local
        elif command == '-mt5' or command == '-local':
            self.mt5Controller.pricesLoader.switchSource(command[1:])

        else:
            return True
