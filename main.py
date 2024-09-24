import sys
# sys.path.append('mt5Mvc')
# sys.path.append('mt5Mvc/mt5Mvc/')

from mt5Mvc.models.myUtils import inputModel
from mt5Mvc.controllers.commands.CommandController import CommandController

commandController = CommandController()
while(True):
    # try:
        command = inputModel.enter()
        commandController.run(command)
    # except:
    #     print('error occurred')
