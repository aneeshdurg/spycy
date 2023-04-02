from typing import List

import pandas as pd


def left(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("left is unimplemented")


def lTrim(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("lTrim is unimplemented")


def replace(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("replace is unimplemented")


def reverse(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("reverse is unimplemented")


def right(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("right is unimplemented")


def rTrim(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("rTrim is unimplemented")


def split(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("split is unimplemented")


def substring(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("substring is unimplemented")


def toLower(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("toLower is unimplemented")


def toString(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("toString is unimplemented")


def toUpper(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("toUpper is unimplemented")


def trim(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("trim is unimplemented")


fn_map = {
    "left": left,
    "lTrim": lTrim,
    "replace": replace,
    "reverse": reverse,
    "right": right,
    "rTrim": rTrim,
    "split": split,
    "substring": substring,
    "toLower": toLower,
    "toString": toString,
    "toUpper": toUpper,
    "trim": trim,
}
