from dataclasses import dataclass
from typing import Generic, List, Union

import pandas as pd

from spycy.graph import EdgeType, Graph, NodeType


@dataclass
class Node(Generic[NodeType]):
    id_: NodeType

    def __lt__(self, other: "Node") -> bool:
        assert isinstance(other, Node)
        return self.id_ < other.id_


@dataclass
class Edge(Generic[EdgeType]):
    id_: EdgeType

    def __lt__(self, other: "Edge") -> bool:
        assert isinstance(other, Edge)
        return self.id_ < other.id_


@dataclass
class Path(Generic[NodeType, EdgeType]):
    nodes: List[NodeType]
    edges: List[Union[EdgeType, List[EdgeType]]]

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
