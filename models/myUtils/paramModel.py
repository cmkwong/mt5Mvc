import inspect
import re
from datetime import date, datetime

from myUtils import fileModel
from dataclasses import dataclass


def paramPreprocess(input_data, param):
    if param.annotation == list:
        required_input_data = input_data.split(' ')
        if len(input_data) == 0: required_input_data = []
    elif param.annotation == tuple:
        required_input_data = eval(input_data)
    elif type(input_data) != param.annotation:
        required_input_data = param.annotation(input_data)  # __new__, refer: https://www.pythontutorial.net/python-oop/python-__new__/
    else:
        required_input_data = input_data
    return required_input_data


def read_default_param(strategy_name, main_path, paramFile):
    text = fileModel.read_text(main_path, paramFile)
    strategy_param_text = [t for t in text.split('~') if len(t) > 0 and t.find(strategy_name) >= 0][0].strip()
    strategy_param_dict = {}
    for param_text in strategy_param_text.split('\n')[1:]:
        param_name, value = param_text.split(':', 1)
        strategy_param_dict[param_name.strip()] = value.strip()
    return strategy_param_dict


def input_param(param, strategy_param_dict):
    input_data = input("{}({})\nDefault: {}: ".format(param.getName, param.annotation.__name__, strategy_param_dict[param.getName]))
    if len(input_data) == 0:
        input_data = strategy_param_dict[param.getName]
    return input_data


def ask_params(class_object, main_path: str, paramFile: str):
    # read the default params text
    strategy_param_dict = read_default_param(class_object.__name__, main_path, paramFile)

    # params details from object
    sig = inspect.signature(class_object)
    params = {}
    for param in sig.parameters.values():
        if (param.kind == param.KEYWORD_ONLY) and (param.default == param.empty):
            input_data = input_param(param, strategy_param_dict)  # asking params
            input_data = paramPreprocess(input_data, param)
            params[param.name] = input_data
    return params


def insert_params(class_object, input_datas: list):
    """
    inputParams and class_object must have same order and length
    :param class_object: class
    :param inputParams: list of input parameters
    :return: {params}
    """
    # params details from object
    sig = inspect.signature(class_object)
    params = {}
    for i, param in enumerate(sig.parameters.values()):
        if (param.kind == param.KEYWORD_ONLY):  # and (param.default == param.empty)
            input_data = paramPreprocess(input_datas[i], param)
            params[param.name] = input_data
    return params


class SymbolList(list):
    @staticmethod
    def get_default_text():
        return """
            EURUSD AUDJPY AUDUSD CADJPY USDCAD 
        """

    @staticmethod
    def __new__(cls, text:str):
        if len(text) == 0:
            text = cls.get_default_text()
        seq = text.strip().split(' ')
        symbols = [str(s) for s in seq]
        return symbols


class DatetimeTuple(object):
    @staticmethod
    def __new__(cls, inputDate):
        """
        :param inputDate: tuple/date
        """
        if isinstance(inputDate, date):
            return (inputDate.year, inputDate.month, inputDate.day, 0, 0) # (year, month, day, hour, minute)
        if isinstance(inputDate, datetime):
            return (inputDate.year, inputDate.month, inputDate.day, inputDate.hour, inputDate.minute) # (year, month, day, hour, minute)
        return inputDate


class InputBoolean(object):
    @staticmethod
    def __new__(cls, text:str):
        boolean = bool(int(text))
        return boolean


class Tech_Dict(object):
    @staticmethod
    def get_default_text():
        return """
            ma 5 10 25 50 100 150 200 250 ;
            bb (20,2,2,0) (20,3,3,0) (20,4,4,0) (40,2,2,0) (40,3,3,0) (40,4,4,0) ;
            std (5,1) (20,1) (50,1) (100,1) (150,1) (250,1) ;
            rsi 5 15 25 50 100 150 250 ;
            stocOsci (5,3,3,0,0) (14,3,3,0,0) (21,14,14,0,0) ;
            macd (12,26,9) (19,39,9) ;
        """

    @staticmethod
    def get_splited_text(text:str):
        splited_text = [t.strip() for t in text.split(';') if len(t.strip()) > 0]

        return splited_text

    @staticmethod
    def text_to_dic(splited_text):
        dic = {}
        for raw_text in splited_text:
            text = raw_text.split(' ', 1)
            dic[text[0]] = text[1]
        return dic

    @staticmethod
    def get_params(param_text):
        # tuple-based params: [(20,2,2,0),(20,3,3,0),(20,4,4,0),(40,2,2,0),(40,3,3,0),(40,4,4,0)]
        regex = r"\(\S+?\)"
        results = re.findall(regex, param_text)
        if results:
            params = []
            for result in results:
                param = [int(r.replace('(', '').replace(')', '')) for r in result.split(',')]
                param = tuple(param)
                params.append(param)
        # list-based params: [5,10,25,50,100,150,200,250]
        else:
            params = [int(p) for p in param_text.split(' ')]
        return params

    @staticmethod
    def __new__(cls, text:str):
        """
        text: ma 5 10 25 50 100 150 200 250; bb (20,2,2,0) (20,3,3,0) (20,4,4,0) (40,2,2,0) (40,3,3,0) (40,4,4,0); std (5,1) (20,1) (50,1) (100,1) (150,1) (250,1); rsi 5 15 25 50 100 150 250; stocOsci (5,3,3,0,0) (14,3,3,0,0) (21,14,14,0,0); macd (12,26,9) (19,39,9)
        tech_params = {
            'ma': [5,10,25,50,100,150,200,250],
            'bb': [(20,2,2,0),(20,3,3,0),(20,4,4,0),(40,2,2,0),(40,3,3,0),(40,4,4,0)],
            'std': [(5,1),(20,1),(50,1),(100,1),(150,1),(250,1)],
            'rsi': [5,15,25,50,100,150,250],
            'stocOsci': [(5,3,3,0,0),(14,3,3,0,0),(21,14,14,0,0)],
            'macd': [(12,26,9),(19,39,9)]
        }
        """
        # if len(text) == 0:
        #     text = cls.get_default_text()
        params = {}
        splited_text = cls.get_splited_text(text)
        raw_dic = cls.text_to_dic(splited_text)
        for k, param_text in raw_dic.items():
            params[k] = cls.get_params(param_text)
        return params