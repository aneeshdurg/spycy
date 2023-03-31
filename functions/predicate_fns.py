from typing import List

import pandas as pd


def exists(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("exists is unimplemented")


def abs_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("abs is unimplemented")


def ceil(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("ceil is unimplemented")


def floor(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("floor is unimplemented")


def rand(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("rand is unimplemented")


def round_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("round is unimplemented")


def sign(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("sign is unimplemented")


fn_map = {
    "abs": abs_,
    "ceil": ceil,
    "floor": floor,
    "rand": rand,
    "round": round_,
    "sign": sign,
}
