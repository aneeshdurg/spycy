from typing import List

import pandas as pd

from spycy.errors import ExecutionError
from spycy.types import FunctionContext


def string_func(f):
    def wrapper(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
        if len(params) != 1:
            raise ExecutionError(f"Invalid number of arguments")

        output = []
        for s in params[0]:
            if s is pd.NA:
                output.append(pd.NA)
            elif not isinstance(s, str):
                raise ExecutionError(f"TypeError::Expected string")
            else:
                output.append(f(s))
        return pd.Series(output, dtype=str)

    return wrapper


def left(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("left is unimplemented")


def replace(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("replace is unimplemented")


def right(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("right is unimplemented")


def split(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    if len(params) != 2:
        raise ExecutionError(f"Invalid number of arguments")
    output = []
    for arg0, arg1 in zip(params[0], params[1]):
        if arg0 is pd.NA or arg1 is pd.NA:
            output.append(pd.NA)
        else:
            if not isinstance(arg0, str) or not isinstance(arg1, str):
                raise ExecutionError(
                    f"TypeError::split expected strings got types: {type(arg0)}, {type(arg1)}"
                )
            output.append(arg0.split(arg1))
    return pd.Series(output)


def substring(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    if len(params) not in [2, 3]:
        raise ExecutionError(f"Invalid number of arguments")

    param2 = params[2] if len(params) == 3 else ([None] * len(params[0]))
    output = []
    for arg0, arg1, end in zip(params[0], params[1], param2):
        if end is pd.NA:
            output.append(pd.NA)
        elif arg0 is pd.NA or arg1 is pd.NA:
            output.append(pd.NA)
        else:
            if not isinstance(arg0, str) or not isinstance(arg1, int):
                raise ExecutionError(
                    f"TypeError::split expected (str, int) got types: {type(arg0)}, {type(arg1)}"
                )
            output.append(arg0[arg1:end])
    return pd.Series(output)


def toString(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("toString is unimplemented")


fn_map = {
    "left": left,
    "ltrim": string_func(lambda x: x.lstrip()),
    "replace": replace,
    "reverse": string_func(lambda x: x[::-1]),
    "right": right,
    "rtrim": string_func(lambda x: x.rstrip()),
    "split": split,
    "substring": substring,
    "tolower": string_func(lambda x: x.lower()),
    "tostring": toString,
    "toupper": string_func(lambda x: x.upper()),
    "trim": string_func(lambda x: x.strip()),
}
