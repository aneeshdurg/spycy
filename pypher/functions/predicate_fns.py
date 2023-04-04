from typing import List

import pandas as pd

from pypher.errors import ExecutionError


def exists(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    if len(params) != 1:
        raise ExecutionError(f"Invalid number of arguments")
    return pd.Series([x is not pd.NA for x in params[0]], dtype=bool)


fn_map = {
    "exists": exists,
}
