from typing import List

import numpy as np
import pandas as pd

from spycy.errors import ExecutionError


def keys(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("keys not implemented")


def labels(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("labels not implemented")


def nodes(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("nodes not implemented")


def range_(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    if len(params) not in [2, 3]:
        raise ExecutionError("Incorrect argument count for range")

    output = []
    for i in range(len(params[0])):
        start = params[0][i]
        end = params[1][i]
        if not np.issubdtype(type(start), np.integer):
            raise ExecutionError(
                f"TypeError::range must take ints as arguments, got {type(start)}"
            )
        if not np.issubdtype(type(end), np.integer):
            raise ExecutionError(
                f"TypeError::range must take ints as arguments, got {type(end)}"
            )
        step = 1
        if len(params) == 3:
            step = params[2][i]
            if step == 0:
                raise ExecutionError("NumberOutOfRange::range's step must be > 0")
            if not np.issubdtype(type(step), np.integer):
                raise ExecutionError(
                    f"TypeError::range must take ints as arguments, got {type(step)}"
                )

        if start == end:
            output.append([start])
        else:
            end += 1 if step > 0 else -1
            output.append(list(range(start, end, step)))
    return pd.Series(output)


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
