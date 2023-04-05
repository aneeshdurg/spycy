from typing import List

import pandas as pd

from . import (
    aggregation_fns,
    list_fns,
    logarithmic_fns,
    numeric_fns,
    predicate_fns,
    scalar_fns,
    string_fns,
    trignometric_fns,
)

fn_maps = [
    predicate_fns.fn_map,
    scalar_fns.fn_map,
    aggregation_fns.fn_map,
    list_fns.fn_map,
    numeric_fns.fn_map,
    logarithmic_fns.fn_map,
    trignometric_fns.fn_map,
    string_fns.fn_map,
]


def function_registry(
    name: str, params: List[pd.Series], table: pd.DataFrame
) -> pd.Series:
    for fn_map in fn_maps:
        if fn := fn_map.get(name, None):
            return fn(params, table)

    raise AssertionError(f"Function {name} not found")


def is_aggregation(name: str):
    return name in aggregation_fns.fn_map
