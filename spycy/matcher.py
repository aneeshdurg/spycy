from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Union

import pandas as pd

import spycy.pattern_graph as pattern_graph
from spycy.gen.CypherParser import CypherParser
from spycy.graph import EdgeType, Graph, NodeType

# Either a single edge, or a variable length relationship
MatchedEdge = Union[EdgeType, List[EdgeType]]


@dataclass
class MatchResult(Generic[NodeType, EdgeType]):
    node_ids_to_data_ids: Dict[pattern_graph.NodeID, NodeType] = field(
        default_factory=dict
    )
    edge_ids_to_data_ids: Dict[pattern_graph.EdgeID, MatchedEdge[EdgeType]] = field(
        default_factory=dict
    )

    def contains_edge(self, edge: MatchedEdge) -> bool:
        if isinstance(edge, list):
            for e in edge:
                if self.contains_edge(e):
                    return True
            return False
        for matched_edge in self.edge_ids_to_data_ids.values():
            if isinstance(matched_edge, list):
                if edge in matched_edge:
                    return True
            else:
                if matched_edge == edge:
                    return True
        return False

    def copy(self) -> "MatchResult":
        return MatchResult(
            self.node_ids_to_data_ids.copy(), self.edge_ids_to_data_ids.copy()
        )


@dataclass
class MatchResultSet(Generic[NodeType, EdgeType]):
    node_ids_to_data_ids: Dict[pattern_graph.NodeID, List[NodeType]] = field(
        default_factory=dict
    )
    edge_ids_to_data_ids: Dict[pattern_graph.EdgeID, List[MatchedEdge]] = field(
        default_factory=dict
    )

    def __len__(self) -> int:
        if len(self.node_ids_to_data_ids) == 0:
            return 0
        return len(next(self.node_ids_to_data_ids.values().__iter__()))

    def add(self, result: MatchResult):
        if len(self.node_ids_to_data_ids) == 0:
            for nid, data in result.node_ids_to_data_ids.items():
                self.node_ids_to_data_ids[nid] = [data]
            for eid, data in result.edge_ids_to_data_ids.items():
                self.edge_ids_to_data_ids[eid] = [data]
        else:
            for nid in self.node_ids_to_data_ids:
                self.node_ids_to_data_ids[nid].append(result.node_ids_to_data_ids[nid])
            for eid in self.edge_ids_to_data_ids:
                self.edge_ids_to_data_ids[eid].append(result.edge_ids_to_data_ids[eid])


class Matcher(Generic[NodeType, EdgeType], metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def match(
        cls,
        graph: Graph[NodeType, EdgeType],
        variables: Dict[str, Any],
        parameters: Dict[str, Any],
        filter_: Optional[CypherParser.OC_WhereContext],
        pgraph: pattern_graph.Graph,
        row_id: int,
        node_ids_to_props: Dict[pattern_graph.NodeID, pd.Series],
        edge_ids_to_props: Dict[pattern_graph.EdgeID, pd.Series],
        initial_matched: MatchResult[NodeType, EdgeType],
        find_at_least_one: bool = False,
    ) -> MatchResultSet[NodeType, EdgeType]:
        pass
