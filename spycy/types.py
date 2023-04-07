from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import pandas as pd

DataEdge = Tuple[int, int, int]


@dataclass
class Node:
    id_: int

    def __lt__(self, other: "Node") -> bool:
        assert isinstance(other, Node)
        return self.id_ < other.id_


DataEdge = Tuple[int, int, int]


@dataclass
class Edge:
    id_: DataEdge

    def __lt__(self, other: "Edge") -> bool:
        assert isinstance(other, Edge)
        return self.id_ < other.id_


@dataclass
class FunctionContext:
    table: pd.DataFrame
    graph: nx.MultiDiGraph
