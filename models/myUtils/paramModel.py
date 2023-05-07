import inspect
import re
from datetime import date, datetime
from models.myUtils import fileModel
from prompt_toolkit import prompt


def decodeParam(input_data, paramSig):
    """
    eg:  "AUDCAD EURUSD AUDUSD" -> ["AUDCAD", "EURUSD", "AUDUSD"]
    """
    if paramSig.annotation == list:
        required_input_data = input_data.split(' ')
        if len(input_data) == 0: required_input_data = []
    elif paramSig.annotation == tuple:
        required_input_data = eval(input_data)
    elif type(input_data) != paramSig.annotation:
        required_input_data = paramSig.annotation(input_data)  # __new__, refer: https://www.pythontutorial.net/python-oop/python-__new__/
    else:
        required_input_data = input_data
    return required_input_data

# convert dictionary parameter into raw string
def encodeParam(dictParam):
    """
    eg: ["AUDCAD", "EURUSD", "AUDUSD"] -> "AUDCAD EURUSD AUDUSD"
    """
    encoded_params = {}
    for key, param in dictParam.items():
        if isinstance(param, list):
            raw_param = " ".join([str(p) for p in param])
        elif isinstance(param, tuple):
            raw_param = str(param)
        else:
            raw_param = str(param)
        encoded_params[key] = raw_param
    return encoded_params

# user input the param
def input_param(paramSig, encoded_params):
    # ask use input parameter and allow user to modify the default parameter
    input_data = prompt(f"{paramSig.name}({paramSig.annotation.__name__}): ", default=encoded_params[paramSig.name])
    # if no input, then assign default parameter
    if len(input_data) == 0:
        input_data = encoded_params[paramSig.name]
    return input_data

# def get_params(class_object, dictParam=None, preprocessNeed=False):
#     if not dictParam:
#         dictParam = {}
#     # params details from object
#     sig = inspect.signature(class_object)
#     params = {}
#     # looping the signature
#     for paramSig in sig.parameters.values():
#         # argument after(*) && has no default argument
#         if paramSig.kind == paramSig.KEYWORD_ONLY:
#             # check if parameter is missed in default and assigned dict
#             if paramSig.name not in dictParam.keys():
#                 if paramSig.default == paramSig.empty:
#                     raise Exception(f'{paramSig.name} parameter is missed. ')
#                 else:
#                     dictParam[paramSig.name] = paramSig.default
#             # encode the param
#             encoded_params = encodeParam(dictParam)
#             # asking params
#             input_data = input_param(paramSig, encoded_params)
#             # preprocess the param
#             if preprocessNeed: input_data = decodeParam(input_data, paramSig)
#             params[paramSig.name] = input_data
#     return params


# # read the param from text
# def get_raw_param_dict_from_txt_file(object_name, main_path, paramFile):
#     text = fileModel.read_text(main_path, paramFile)
#     param_text = [t for t in text.split('~') if len(t) > 0 and t.find(object_name) >= 0][0].strip()
#     param_dict = {}
#     for param_text in param_text.split('\n')[1:]:
#         param_name, value = param_text.split(':', 1)
#         param_dict[param_name.strip()] = value.strip()
#     return param_dict
#
#
# # ask user to input parameter or read from txt source
# def ask_txtParams(class_object, main_path='', paramFile=''):
#     # read the default params text
#     encoded_params = get_raw_param_dict_from_txt_file(class_object.__name__, main_path, paramFile)
#     params = get_params(class_object, encoded_params, True)
#     return params

# ask user to input parameter from dictionary
def ask_params(class_object, dictParam=None, preprocessNeed=False):
    if not dictParam:
        dictParam = {}
    # params details from object
    sig = inspect.signature(class_object)
    params = {}
    # looping the signature
    for paramSig in sig.parameters.values():
        # argument after(*) && has no default argument
        if paramSig.kind == paramSig.KEYWORD_ONLY:
            # check if parameter is missed in default and assigned dict
            if paramSig.name not in dictParam.keys():
                if paramSig.default == paramSig.empty:
                    raise Exception(f'{paramSig.name} parameter is missed. ')
                else:
                    dictParam[paramSig.name] = paramSig.default
            # encode the param
            encoded_params = encodeParam(dictParam)
            # asking params
            input_data = input_param(paramSig, encoded_params)
            # preprocess the param
            if preprocessNeed: input_data = decodeParam(input_data, paramSig)
            params[paramSig.name] = input_data
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
    for i, paramSig in enumerate(sig.parameters.values()):
        if (paramSig.kind == paramSig.KEYWORD_ONLY):  # and (param.default == param.empty)
            input_data = decodeParam(input_datas[i], paramSig)
            params[paramSig.name] = input_data
    return params


class SymbolList(list):
    @staticmethod
    def get_default_text():
        return """
            EURUSD AUDJPY AUDUSD CADJPY USDCAD 
        """

    @staticmethod
    def __new__(cls, text: str):
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
