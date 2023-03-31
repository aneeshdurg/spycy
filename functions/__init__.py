from typing import List

import pandas as pd

from . import aggregation_fns
from . import list_fns
from . import logarithmic_fns
from . import numeric_fns
from . import predicate_fns
from . import scalar_fns
from . import string_fns
from . import trignometric_fns

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

def function_registry(name: str, params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    for fn_map in fn_maps:
        if fn := fn_map.get(name, None):
            return fn(params, table)

    raise AssertionError(f"Function {name} not found")
