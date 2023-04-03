from typing import List

import pandas as pd

from pypher.errors import ExecutionError


def coalesce(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    data = []
    for i in range(len(params[0])):
        for param in params:
            if param[i] is not pd.NA:
                data.append(param[i])
                break
    return pd.Series(data)


def endNode(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("endNode unimplemented")


def head(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("head unimplemented")


def id_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("id unimplemented")


def last(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("last unimplemented")


def length(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("length unimplemented")


def properties(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("properties unimplemented")


def size(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("size unimplemented")


def startNode(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("startNode unimplemented")


def timestamp(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("timestamp unimplemented")


def toBoolean(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    if len(params) > 1:
        raise ExecutionError("Invalid number of arguments to toBoolean")
    arg = params[0]
    output = []
    for i in range(len(arg)):
        if arg[i] is pd.NA:
            output.append(pd.NA)
        elif arg[i] == True or arg[i] == False:
            # could be numpy.bool_ or bool
            output.append(bool(arg[i]))
        elif isinstance(arg[i], str):
            if arg[i] == "true":
                output.append(True)
            elif arg[i] == "false":
                output.append(False)
            else:
                output.append(pd.NA)
        else:
            raise ExecutionError("TypeError::toBoolean got unexpected type")
    return pd.Series(output)


def toFloat(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("toFloat unimplemented")


def toInteger(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    if len(params) > 1:
        raise ExecutionError("Invalid number of arguments to toBoolean")
    arg = params[0]
    output = []
    for i in range(len(arg)):
        val = arg[i]
        if val is pd.NA:
            output.append(pd.NA)
        elif isinstance(val, float):
            output.append(int(val))
        elif isinstance(val, str):
            output.append(pd.NA)
        else:
            raise ExecutionError("TypeError::toInteger got unexpected type")
    return pd.Series(output)


def type_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("type unimplemented")


fn_map = {
    "coalesce": coalesce,
    "endNode": endNode,
    "head": head,
    "id": id_,
    "last": last,
    "length": length,
    "properties": properties,
    "size": size,
    "startNode": startNode,
    "timestamp": timestamp,
    "toBoolean": toBoolean,
    "toFloat": toFloat,
    "toInteger": toInteger,
    "type": type_,
}
