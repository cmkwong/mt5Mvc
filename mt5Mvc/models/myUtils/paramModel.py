from mt5Mvc.models.myUtils import inputModel

import inspect
import re
from datetime import date, datetime
from prompt_toolkit import prompt

def decodeParam(input_data, dataType):
    """
    list:   ["AUDCAD", "EURUSD", "AUDUSD"] -> "AUDCAD EURUSD AUDUSD"
    tuple:  ("AUDCAD", "EURUSD", "AUDUSD") -> '("AUDCAD", "EURUSD", "AUDUSD")'
    other:  1 -> '1'
    """
    if dataType == any:
        required_input_data = eval(input_data)
    elif dataType == list:
        required_input_data = input_data.split(' ')
        if len(input_data) == 0:
            required_input_data = []
    elif dataType == tuple:
        required_input_data = eval(input_data)
    elif dataType == bool:
        required_input_data = False
        if input_data.upper() == 'TRUE':
            required_input_data = True
    # as space/empty cannot int / float the transform
    elif dataType in (int, float) and (input_data == '' or input_data.isspace()):
        required_input_data = 0
    elif type(input_data) != dataType:
        required_input_data = dataType(input_data)  # __new__, refer: https://www.pythontutorial.net/python-oop/python-__new__/
    else:
        required_input_data = input_data
    return required_input_data

# convert dictionary parameter into raw string
def encodeParam(param):
    """
    list:   ["AUDCAD", "EURUSD", "AUDUSD"] -> "AUDCAD EURUSD AUDUSD"
    tuple:  ("AUDCAD", "EURUSD", "AUDUSD") -> '("AUDCAD", "EURUSD", "AUDUSD")'
    other:  1 -> '1'
    """
    if isinstance(param, list):
        encoded_param = " ".join([str(p) for p in param])
    elif isinstance(param, tuple):
        encoded_param = str(param)
    else:
        encoded_param = str(param)
    return encoded_param

# user input the param
def input_param(paramName, paramValue, dataTypeName):
    # ask use input parameter and allow user to modify the default parameter
    input_data = prompt(f"{paramName}({dataTypeName}): ", default=paramValue)
    # if no input, then assign default parameter
    if len(input_data) == 0:
        input_data = paramValue
    return input_data

# # ask user to input parameter from dictionary
# def ask_params_DISCARD(class_object, **kwargs):
#     """
#     :param class_object: class / function attribute
#     :param kwargs: dict
#     :return:
#     """
#     # if it is none
#     if not kwargs: kwargs = {}
#     # params details from object
#     signatures = inspect.signature(class_object)
#     params = {}
#     # looping the signature
#     for sig in signatures.parameters.values():
#         # argument after(*)
#         if sig.kind == sig.KEYWORD_ONLY:
#             # encode the param
#             if sig.name in kwargs.keys():
#                 encoded_param = encodeParam(kwargs[sig.name])
#             else:
#                 # has no default parameter
#                 if sig.default == sig.empty:
#                     encoded_param = ''
#                 else:
#                     encoded_param = encodeParam(sig.default)
#             # asking params
#             input_data = input_param(sig.name, encoded_param, sig.annotation.__name__)
#             # preprocess the param
#             input_data = decodeParam(input_data, sig.annotation)
#             params[sig.name] = input_data
#     return class_object, params

def ask_param_fn(class_object, **overwrote_paramFormat):
    """
    :param class_object: class / function attribute
    :param overwrote_paramFormat: dict
    :return: obj, dict of param
    """
    # if it is none
    if not overwrote_paramFormat: overwrote_paramFormat = {}
    # params details from object
    signatures = inspect.signature(class_object)
    paramFormat = {}
    # looping the signature
    for sig in signatures.parameters.values():
        # argument after(*)
        if sig.kind == sig.KEYWORD_ONLY:
            # encode the param
            if sig.name in overwrote_paramFormat.keys():
                paramFormat[sig.name] = overwrote_paramFormat[sig.name]
            else:
                # has no default parameter
                if sig.default == sig.empty:
                    paramFormat[sig.name] = ['', sig.annotation, 'field']
                else:
                    paramFormat[sig.name] = [sig.default, sig.annotation, 'field']
    # ask user to input the param
    param = ask_param(paramFormat)
    return class_object, param

def ask_param(paramFormat):
    """
    purely to ask the param base on the dictionary
    :param paramFormat: dict, { name: [value, dataType, fieldType] }
    fieldType = ['field', 'options']
    :return:
    """
    params = {}
    for name, (value, dataType, fieldType) in paramFormat.items():
        if fieldType == 'field':
            # encode the param (for user input), eg: [a, b, c] => "a b c"
            encoded_param = encodeParam(value)
            # asking params
            input_data = input_param(name, encoded_param, dataType.__name__)
            # decode the param
            decode_data = decodeParam(input_data, dataType)
            params[name] = decode_data
        elif fieldType == 'dropdown':
            params[name] = decodeParam(value[inputModel.askSelection(value, f"{name} ({dataType})\nPlease Select")], dataType)
        else:
            raise Exception('Please specify the field type. ')
    return params
# def insert_params(class_object, input_datas: list):
#     """
#     inputParams and class_object must have same order and length
#     :param class_object: class
#     :param inputParams: list of input parameters
#     :return: {params}
#     """
#     # params details from object
#     sig = inspect.signature(class_object)
#     params = {}
#     for i, paramSig in enumerate(sig.parameters.values()):
#         if (paramSig.kind == paramSig.KEYWORD_ONLY):  # and (param.default == param.empty)
#             input_data = decodeParam(input_datas[i], paramSig)
#             params[paramSig.name] = input_data
#     return params


class SymbolList(list):
    """
    Can allowed to be string with space and a list of value
    """
    @staticmethod
    def get_default_text():
        return """
            EURUSD AUDJPY AUDUSD CADJPY USDCAD 
        """

    @staticmethod
    def __new__(cls, symbols):
        if isinstance(symbols, list):
            return symbols
        elif isinstance(symbols, str):
            if len(symbols) == 0:
                symbols = cls.get_default_text()
            symbol = symbols.strip().split(' ')
            symbols = [str(s) for s in symbol]
        return symbols


class DatetimeTuple(object):
    @staticmethod
    def __new__(cls, inputDate):
        """
        :param inputDate: tuple/date
        """
        if isinstance(inputDate, date):
            return (inputDate.year, inputDate.month, inputDate.day, 0, 0)  # (year, month, day, hour, minute)
        if isinstance(inputDate, datetime):
            return (inputDate.year, inputDate.month, inputDate.day, inputDate.hour, inputDate.minute)  # (year, month, day, hour, minute)
        if isinstance(inputDate, str):
            rawTuple = eval(inputDate)
            dateTimeList = [0, 0, 0, 0, 0]
            for i, el in enumerate(rawTuple):
                dateTimeList[i] = int(el)
            return tuple(dateTimeList)
        return inputDate


class InputBoolean(object):
    @staticmethod
    def __new__(cls, text: str):
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
    def get_splited_text(text: str):
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
    def __new__(cls, text: str):
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
