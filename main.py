import sys
sys.path.append('C:/Users/Chris/projects/221227_mt5Mvc')
sys.path.append('C:/Users/Chris/projects/AtomLib')

from models.myUtils import inputModel
from controllers.commands.CommandController import CommandController

commandController = CommandController()
while(True):
    # try:
        command = inputModel.enter()
        commandController.run(command)
    # except:
    #     print('error occurred')
