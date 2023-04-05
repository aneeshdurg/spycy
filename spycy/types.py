from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import pandas as pd


@dataclass
class Node:
    id_: int

    def __lt__(self, other: "Node") -> bool:
        assert isinstance(other, Node)
        return self.id_ < other.id_


@dataclass
class Edge:
    id_: Tuple[int, int, int]

    def __lt__(self, other: "Edge") -> bool:
        assert isinstance(other, Edge)
        return self.id_ < other.id_


@dataclass
class FunctionContext:
    table: pd.DataFrame
    graph: nx.MultiDiGraph
