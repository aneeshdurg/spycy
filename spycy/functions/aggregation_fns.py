from typing import List

import numpy as np
import pandas as pd

from spycy.errors import ExecutionError
from spycy.types import FunctionContext


def agg_func(f):
    def wrapper(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
        if len(params) != 1:
            raise ExecutionError(f"Invalid number of arguments")

        return pd.Series([f(params[0])])

    return wrapper


def percentileCont(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("percentileCont not implemented")


def percentileDisc(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("percentileDisc not implemented")


def stDev(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("stDev not implemented")


def stDevP(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("stDevP not implemented")


fn_map = {
    "avg": agg_func(np.mean),
    "collect": agg_func(list),
    "count": agg_func(lambda f: np.sum(f.apply(lambda e: e is not pd.NA))),
    "max": agg_func(np.max),
    "min": agg_func(np.min),
    "percentileCont": percentileCont,
    "percentileDisc": percentileDisc,
    "stDev": stDev,
    "stDevP": stDevP,
    "sum": agg_func(np.sum),
}
