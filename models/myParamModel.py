from controllers.strategies.SwingScalping.Live import Live as SwingScalping_Live
from controllers.strategies.SwingScalping.Backtest import Backtest as SwingScalping_Backtest
from myUtils import fileModel
from controllers.strategies.RL.Train import Train as RL_Train

# STRATEGY_PARAMS { Strategy name: [ { 'base', 'run' }, { 'base', 'run' }, ... ] }
STRATEGY_PARAMS = {
    'live': {
        fileModel.getParentFolderName(SwingScalping_Live): [

            {'base': {'symbol': 'GBPUSD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 70, 'diff_ema_middle_lower': 60, 'ratio_sl_sp': 2.2,
                 'lowerEma': 18, 'middleEma': 55, 'upperEma': 92,
                 'trendType': 'rise', 'lot': 2,
             }},
            {'base': {'symbol': 'GBPUSD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 70, 'diff_ema_middle_lower': 60, 'ratio_sl_sp': 2.2,
                 'lowerEma': 18, 'middleEma': 55, 'upperEma': 92,
                 'trendType': 'down', 'lot': 2,
             }},
            {'base': {'symbol': 'USDJPY', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 70, 'diff_ema_middle_lower': 60, 'ratio_sl_sp': 2.2,
                 'lowerEma': 18, 'middleEma': 55, 'upperEma': 92,
                 'trendType': 'rise', 'lot': 2,
             }},
            {'base': {'symbol': 'USDJPY', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 70, 'diff_ema_middle_lower': 60, 'ratio_sl_sp': 2.2,
                 'lowerEma': 18, 'middleEma': 55, 'upperEma': 92,
                 'trendType': 'down', 'lot': 2,
             }},
            {'base': {'symbol': 'AUDJPY', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 40, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 2,
                 'lowerEma': 18, 'middleEma': 39, 'upperEma': 96,
                 'trendType': 'rise', 'lot': 2,
             }},
            {'base': {'symbol': 'AUDJPY', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 30, 'diff_ema_middle_lower': 40, 'ratio_sl_sp': 2.2,
                 'lowerEma': 26, 'middleEma': 47, 'upperEma': 88,
                 'trendType': 'down', 'lot': 2,
             }},
            {'base': {'symbol': 'CADJPY', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 70, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 2,
                 'lowerEma': 18, 'middleEma': 31, 'upperEma': 84,
                 'trendType': 'rise', 'lot': 2,
             }},
            {'base': {'symbol': 'CADJPY', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 20, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 1.8,
                 'lowerEma': 18, 'middleEma': 47, 'upperEma': 92,
                 'trendType': 'down', 'lot': 2,
             }},
            {'base': {'symbol': 'AUDUSD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 70, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 2,
                 'lowerEma': 18, 'middleEma': 31, 'upperEma': 84,
                 'trendType': 'rise', 'lot': 2,
             }},
            {'base': {'symbol': 'AUDUSD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 50, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 1.8,
                 'lowerEma': 18, 'middleEma': 27, 'upperEma': 56,
                 'trendType': 'down', 'lot': 2,
             }},
            {'base': {'symbol': 'USDCAD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 50, 'diff_ema_middle_lower': 30, 'ratio_sl_sp': 1.2,
                 'lowerEma': 18, 'middleEma': 27, 'upperEma': 48,
                 'trendType': 'rise', 'lot': 2,
             }},
            {'base': {'symbol': 'USDCAD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 60, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 2.2,
                 'lowerEma': 18, 'middleEma': 27, 'upperEma': 68,
                 'trendType': 'down', 'lot': 2,
             }},
            {'base': {'symbol': 'EURUSD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 40, 'diff_ema_middle_lower': 30, 'ratio_sl_sp': 1.4,
                 'lowerEma': 18, 'middleEma': 35, 'upperEma': 80,
                 'trendType': 'rise', 'lot': 2,
             }},
            {'base': {'symbol': 'EURUSD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 60, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 2.2,
                 'lowerEma': 18, 'middleEma': 31, 'upperEma': 88,
                 'trendType': 'down', 'lot': 2,
             }},
            {'base': {'symbol': 'EURCAD', 'auto': True},
             'run': {
                 'diff_ema_upper_middle': 70, 'diff_ema_middle_lower': 20, 'ratio_sl_sp': 4.6,
                 'lowerEma': 26, 'middleEma': 51, 'upperEma': 92,
                 'trendType': 'down', 'lot': 2,
             }},
        ],
    },
    'train': {
        fileModel.getParentFolderName(RL_Train): [
            {'base': {}, 'run': {}}
        ]
    },
    'backtest': {
        fileModel.getParentFolderName(SwingScalping_Backtest): [
            {'base': {'symbol': ''}}
        ]
    },
    # Train_SwingScalping ...
}


def getParamTxt(strategyName, strategyType):
    """
    :param strategyName: str, class object name
    :param strategyType: 'live' / 'train' / 'backtest'
    :return: txt
    """
    paramTxt = ''
    for i, mainObj in enumerate(STRATEGY_PARAMS[strategyType][strategyName]):
        rowTxt = f'{i}: '
        for setType, param in mainObj.items():  # setType: 'base' / 'run'
            rowTxt += f"({setType}) - {param}; "
        rowTxt += ';\n'
        paramTxt += rowTxt
    return paramTxt


def getParamDic(strategyName, strategyType, paramId):
    """
    :param strategyName: class object name
    :param strategyType: 'live' / 'train' / 'backtest'
    :param paramId: parameter id
    :return: parameter dict
    """
    param = STRATEGY_PARAMS[strategyType][strategyName][int(paramId)]
    return param['base'], param['run']
