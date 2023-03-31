from typing import List

import pandas as pd

def keys(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("keys not implemented")

def labels(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("labels not implemented")

def nodes(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("nodes not implemented")

def range_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("range not implemented")

def relationships(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("relationships not implemented")

def reverse(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("reverse not implemented")

def tail(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("tail not implemented")

fn_map = {
        "keys": keys,
        "labels": labels,
        "nodes": nodes,
        "range": range_,
        "relationships": relationships,
        "reverse": reverse,
        "tail": tail,
}


