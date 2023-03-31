from typing import List

import pandas as pd

def fn(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("fn is unimplemented")


def e_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("e is unimplemented")

def exp(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("exp is unimplemented")

def log(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("log is unimplemented")

def log10(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("log10 is unimplemented")

def sqrt(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("sqrt is unimplemented")

fn_map = {
    "e": e_,
    "exp": exp,
    "log": log,
    "log10": log10,
    "sqrt": sqrt,
}
