from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Mapping, Set, Tuple, TypeVar

import networkx as nx

NodeType = TypeVar("NodeType")
EdgeType = TypeVar("EdgeType")


class Graph(Generic[NodeType, EdgeType], metaclass=ABCMeta):
    @property
    @abstractmethod
    def nodes(self) -> Mapping[NodeType, Any]:
        pass

    @property
    @abstractmethod
    def edges(self) -> Mapping[EdgeType, Any]:
        pass

    @abstractmethod
    def add_node(self, data: Dict[str, Any]) -> NodeType:
        pass

    @abstractmethod
    def add_edge(
        self, start: NodeType, end: NodeType, data: Dict[str, Any]
    ) -> EdgeType:
        pass

    @abstractmethod
    def out_edges(self, node: NodeType) -> List[EdgeType]:
        pass

    @abstractmethod
    def in_edges(self, node: NodeType) -> List[EdgeType]:
        pass

    @abstractmethod
    def remove_node(self, node: NodeType):
        pass

    @abstractmethod
    def remove_edge(self, edge: EdgeType):
        pass

    @abstractmethod
    def src(self, edge: EdgeType) -> NodeType:
        pass

    @abstractmethod
    def dst(self, edge: EdgeType) -> NodeType:
        pass


NetworkXNode = int
NetworkXEdge = Tuple[int, int, int]


@dataclass
class NetworkXGraph(Graph[NetworkXNode, NetworkXEdge]):
    _graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    _deleted_ids: Set[NetworkXNode] = field(default_factory=set)

    def _vend_node_id(self) -> NetworkXNode:
        if len(self._deleted_ids):
            return self._deleted_ids.pop()
        return len(self._graph.nodes)

    @property
    def nodes(self) -> Mapping[NetworkXNode, Any]:
        return self._graph.nodes

    @property
    def edges(self) -> Mapping[NetworkXEdge, Any]:
        return self._graph.edges

    def add_node(self, data: Dict[str, Any]) -> NetworkXNode:
        node_id = self._vend_node_id()
        self._graph.add_node(node_id, **data)
        return node_id

    def add_edge(
        self, start: NetworkXNode, end: NetworkXNode, data: Dict[str, Any]
    ) -> NetworkXEdge:
        key = self._graph.add_edge(start, end, **data)
        return (start, end, key)

    def out_edges(self, node: NetworkXNode) -> List[NetworkXEdge]:
        return list(self._graph.out_edges(node, keys=True))

    def in_edges(self, node: NetworkXNode) -> List[NetworkXEdge]:
        return list(self._graph.in_edges(node, keys=True))

    def remove_node(self, node: NetworkXNode):
        self._graph.remove_node(node)
        self._deleted_ids.add(node)

    def remove_edge(self, edge: NetworkXEdge):
        self._graph.remove_edge(*edge)

    def src(self, edge: NetworkXEdge) -> NetworkXNode:
        return edge[0]

    def dst(self, edge: NetworkXEdge) -> NetworkXNode:
        return edge[1]
