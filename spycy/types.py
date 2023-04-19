from dataclasses import dataclass
from typing import List, Tuple, Union

import pandas as pd

from spycy.graph import DataEdge, DataNode, Graph


@dataclass
class Node:
    id_: DataNode

    def __lt__(self, other: "Node") -> bool:
        assert isinstance(other, Node)
        return self.id_ < other.id_


@dataclass
class Edge:
    id_: DataEdge

    def __lt__(self, other: "Edge") -> bool:
        assert isinstance(other, Edge)
        return self.id_ < other.id_


@dataclass
class Path:
    nodes: List[int]
    edges: List[Union[DataEdge, List[DataEdge]]]

    def __lt__(self, other: "Path") -> bool:
        assert isinstance(other, Edge)
        if self.nodes < other.nodes:
            return True

        if self.nodes > other.nodes:
            return True

        return self.edges < other.edges


@dataclass
class FunctionContext:
    table: pd.DataFrame
    graph: Graph
