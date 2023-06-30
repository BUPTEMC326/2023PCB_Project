# -*- coding:utf-8 -*-
# cql语句
cql={
    "IO": {
        "HDMI":
            {"cql_template" : "match(n) where n.name='HDMI'  return n",
             "prop1":"f_ESDAndFilter",


             },
        "Camera":
            {"cql_template":"match(n) where n.name='Camera'  return n",
             "prop1":"f_ESDAndFilter"},
        "Button":
            {"cql_template":"match(n) where n.name='Button'  return n",
             "prop1":"f_ESDAndFilter"},
    },
    "IC":{
        "IC":
            {"cql_template":"match(n) where n.name='IC'  return n",
             "NCPIN":"NCPIN"},
    },
    "HighSpeedComp":{
        "CLK":
            {"cql_template":"match(n) where n.name='CLK'  return n",
             "model1Radiate":"calcuRadiation"},
        "SwitchPower":
            {"cql_template": "match(n) where n.name='SwitchPower'  return n",
             "model1Radiate":"calcuRadiation"},
    },
    "TranLine":{
            "MicrostripLine":
                {"cql_template":"match(n) where n.name='MicrostripLine'  return n",
                 'diffRadiation':'calcuDiffRadiation'
                 },
        },

}


import functools
import inspect


# 输入类型检查函数
def auto_type_checker(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # fetch the argument name list.
        parameters = inspect.signature(function).parameters
        argument_list = list(parameters.keys())

        # fetch the argument checker list.
        checker_list = [parameters[argument].annotation for argument in argument_list]

        # fetch the value list.
        value_list =  [inspect.getcallargs(function, *args, **kwargs)[argument] for argument in inspect.getfullargspec(function).args]

        # initialize the result dictionary, where key is argument, value is the checker result.
        result_dictionary = dict()
        for argument, value, checker in zip(argument_list, value_list, checker_list):
            result_dictionary[argument] = check(argument, value, checker, function)

        # fetch the invalid argument list.
        invalid_argument_list = [key for key in argument_list if not result_dictionary[key]]

        # if there are invalid arguments, raise the error.
        if len(invalid_argument_list) > 0:
            raise Exception(invalid_argument_list)

        # check the result.
        result = function(*args, **kwargs)
        checker = inspect.signature(function).return_annotation
        if not check('return', result, checker, function):
            raise Exception(['return'])

        # return the result.
        return result
    return wrapper

def check(name, value, checker, function):
    if isinstance(checker, (tuple, list, set)):
        return True in [check(name, value, sub_checker, function) for sub_checker in checker]
    elif checker is inspect._empty:
        return True
    elif checker is None:
        return value is None
    elif isinstance(checker, type):
        return isinstance(value, checker)
    elif callable(checker):
        result = checker(value)
        return result






