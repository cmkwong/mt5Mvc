import os
from models.myUtils import sysModel

PARAM_PATH = './'
PARAM_FILE = 'param.txt'

# relative path for different computer
PROJECT_PATH = sysModel.find_required_path(os.getcwd(), '221227_mt5Mvc')