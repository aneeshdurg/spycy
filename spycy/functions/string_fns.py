from typing import List

import pandas as pd

from spycy.errors import ExecutionError
from spycy.types import FunctionContext


def string_func(f):
    def wrapper(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
        if len(params) != 1:
            raise ExecutionError(f"Invalid number of arguments")
        if not isinstance(params[0][0], str):
            raise ExecutionError(f"TypeError::Expected string")

        return params[0].apply(f)

    return wrapper


def left(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("left is unimplemented")


def replace(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("replace is unimplemented")


def right(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("right is unimplemented")


def split(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("split is unimplemented")


def substring(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("substring is unimplemented")


def toString(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("toString is unimplemented")


fn_map = {
    "left": left,
    "lTrim": string_func(lambda x: x.lstrip()),
    "replace": replace,
    "reverse": string_func(lambda x: x[::-1]),
    "right": right,
    "rTrim": string_func(lambda x: x.rstrip()),
    "split": split,
    "substring": substring,
    "toLower": string_func(lambda x: x.lower()),
    "toString": toString,
    "toUpper": string_func(lambda x: x.upper()),
    "trim": string_func(lambda x: x.strip()),
}
