from typing import List

import pandas as pd

def acos(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("acos is unimplemented")


def asin(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("asin is unimplemented")


def atan(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("atan is unimplemented")


def atan2(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("atan2 is unimplemented")


def cos(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("cos is unimplemented")


def cot(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("cot is unimplemented")


def degrees(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("degrees is unimplemented")


def pi(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("pi is unimplemented")


def radians(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("radians is unimplemented")


def sin(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("sin is unimplemented")


def tan(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("tan is unimplemented")

fn_map = {
        "acos": acos,
        "asin": asin,
        "atan": atan,
        "atan2": atan2,
        "cos": cos,
        "cot": cot,
        "degrees": degrees,
        "pi": pi,
        "radians": radians,
        "sin": sin,
        "tan": tan,
}
