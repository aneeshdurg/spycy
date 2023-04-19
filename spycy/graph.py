from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Set, Tuple

import networkx as nx

DataNode = int
DataEdge = Tuple[int, int, int]


class Graph(metaclass=ABCMeta):
    @property
    @abstractmethod
    def nodes(self) -> Mapping[DataNode, Any]:
        pass

    @property
    @abstractmethod
    def edges(self) -> Mapping[DataEdge, Any]:
        pass

    @abstractmethod
    def add_node(self, data: Dict[str, Any]) -> DataNode:
        pass

    @abstractmethod
    def add_edge(
        self, start: DataNode, end: DataNode, data: Dict[str, Any]
    ) -> DataEdge:
        pass

    @abstractmethod
    def out_edges(self, node: DataNode) -> List[DataEdge]:
        pass

    @abstractmethod
    def in_edges(self, node: DataNode) -> List[DataEdge]:
        pass

    @abstractmethod
    def remove_node(self, node: DataNode):
        pass

    @abstractmethod
    def remove_edge(self, edge: DataEdge):
        pass


@dataclass
class NetworkxGraph(Graph):
    _graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    _deleted_ids: Set[DataNode] = field(default_factory=set)

    def _vend_node_id(self) -> DataNode:
        if len(self._deleted_ids):
            return self._deleted_ids.pop()
        return len(self._graph.nodes)

    @property
    def nodes(self) -> Mapping[DataNode, Any]:
        return self._graph.nodes

    @property
    def edges(self) -> Mapping[DataEdge, Any]:
        return self._graph.edges

    def add_node(self, data: Dict[str, Any]) -> DataNode:
        node_id = self._vend_node_id()
        self._graph.add_node(node_id, **data)
        return node_id

    def add_edge(
        self, start: DataNode, end: DataNode, data: Dict[str, Any]
    ) -> DataEdge:
        key = self._graph.add_edge(start, end, **data)
        return (start, end, key)

    def out_edges(self, node: DataNode) -> List[DataEdge]:
        return list(self._graph.out_edges(node, keys=True))

    def in_edges(self, node: DataNode) -> List[DataEdge]:
        return list(self._graph.in_edges(node, keys=True))

    def remove_node(self, node: DataNode):
        self._graph.remove_node(node)
        self._deleted_ids.add(node)

    def remove_edge(self, edge: DataEdge):
        self._graph.remove_edge(*edge)
