from typing import List

import pandas as pd


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
    raise AssertionError("toBoolean unimplemented")


def toFloat(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("toFloat unimplemented")


def toInteger(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("toInteger unimplemented")


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
