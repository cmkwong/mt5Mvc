
from myUtils import inputModel
from controllers.MainController import MainController
from controllers.CommandController import CommandController

mainController = MainController()
commandController = CommandController(mainController)
while(True):
    command = inputModel.enter()
    commandController.run(command)
