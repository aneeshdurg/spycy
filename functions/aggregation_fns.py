from typing import List

import pandas as pd


def avg(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("avg not implemented")


def collect(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("collect not implemented")


def count(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("count not implemented")


def max_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("max not implemented")


def min_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("min not implemented")


def percentileCont(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("percentileCont not implemented")


def percentileDisc(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("percentileDisc not implemented")


def stDev(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("stDev not implemented")


def stDevP(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("stDevP not implemented")


def sum_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("sum not implemented")


fn_map = {
    "avg": avg,
    "colect": collect,
    "count": count,
    "max": max_,
    "min": min_,
    "percentileCont": percentileCont,
    "percentileDisc": percentileDisc,
    "stDev": stDev,
    "stDevP": stDevP,
    "sum": sum_,
}
