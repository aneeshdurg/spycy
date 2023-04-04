from typing import List

import pandas as pd


def exists(params: List[pd.Series], table: pd.DataFrame) -> pd.Series:
    raise AssertionError("exists is unimplemented")


fn_map = {
    "exists": exists,
}
