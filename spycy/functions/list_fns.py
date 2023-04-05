from typing import List

import numpy as np
import pandas as pd

from spycy.errors import ExecutionError
from spycy.types import FunctionContext, Node


def keys(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("keys not implemented")


def labels(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    if len(params) > 1:
        raise ExecutionError("Invalid number of arguments to labels")

    output = []
    for node in params[0]:
        if not isinstance(node, Node):
            raise ExecutionError("TypeError - labels expects a Node argument")
        node_data = fnctx.graph.nodes[node.id_]
        output.append(sorted(list(node_data["labels"])))
    return pd.Series(output)


def nodes(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("nodes not implemented")


def range_(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
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


def relationships(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("relationships not implemented")


def reverse(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
    raise AssertionError("reverse not implemented")


def tail(params: List[pd.Series], fnctx: FunctionContext) -> pd.Series:
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
