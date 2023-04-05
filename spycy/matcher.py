from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

import spycy.pattern_graph as pattern_graph


@dataclass
class MatchResult:
    node_ids_to_data_ids: Dict[pattern_graph.NodeID, int] = field(default_factory=dict)
    edge_ids_to_data_ids: Dict[pattern_graph.EdgeID, Tuple[int, int, int]] = field(
        default_factory=dict
    )

    def contains_edge(self, edge: Tuple[int, int, int]) -> bool:
        return any(edge in edges for edges in self.edge_ids_to_data_ids.values())

    def copy(self) -> "MatchResult":
        return MatchResult(
            self.node_ids_to_data_ids.copy(), self.edge_ids_to_data_ids.copy()
        )


@dataclass
class MatchResultSet:
    node_ids_to_data_ids: Dict[pattern_graph.NodeID, List[int]] = field(
        default_factory=dict
    )
    edge_ids_to_data_ids: Dict[
        pattern_graph.EdgeID, List[Tuple[int, int, int]]
    ] = field(default_factory=dict)

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


@dataclass
class Matcher:
    graph: nx.MultiDiGraph
    pgraph: pattern_graph.Graph
    row_id: int
    node_ids_to_props: Dict[pattern_graph.NodeID, pd.Series]
    edge_ids_to_props: Dict[pattern_graph.EdgeID, pd.Series]

    results: MatchResultSet = field(default_factory=MatchResultSet)

    def properties_match(
        self, match_props: Dict[str, Any], data_props: Dict[str, Any]
    ) -> bool:
        for k, v in match_props.items():
            if k not in data_props:
                return False
            if v != data_props[k]:
                return False
        return True

    def node_matches(self, pnode: pattern_graph.Node, data_node: int) -> bool:
        if not pnode.labels and not pnode.properties:
            return True

        node_data = self.graph.nodes[data_node]
        if pnode.labels:
            if not pnode.labels <= set(node_data["labels"]):
                return False
        if pnode.properties:
            match_props = self.node_ids_to_props[pnode.id_][self.row_id]
            assert isinstance(match_props, dict)
            data_props = node_data["properties"]
            assert isinstance(data_props, dict)
            if not self.properties_match(match_props, data_props):
                return False
        return True

    def edge_matches(
        self, pedge: pattern_graph.Edge, data_edge: Tuple[int, int, int]
    ) -> bool:
        assert pedge.range_ is None
        if not pedge.types and not pedge.properties:
            return True

        edge_data = self.graph.edges[data_edge]
        if pedge.types:
            if edge_data["type"] not in pedge.types:
                return False
        if pedge.properties:
            match_props = self.edge_ids_to_props[pedge.id_][self.row_id]
            assert isinstance(match_props, dict)
            data_props = edge_data["properties"]
            assert isinstance(data_props, dict)
            if not self.properties_match(match_props, data_props):
                return False
        return True

    def find_all_edges(
        self, source: int, dst: int, pedge: pattern_graph.Edge
    ) -> List[Tuple[int, int, int]]:
        output = []
        for edge in self.graph.out_edges(source, keys=True):
            if self.edge_matches(pedge, edge):
                output.append(edge)
        return output

    def satisfies_edges(
        self,
        intermediate: MatchResult,
        nid: pattern_graph.NodeID,
        extended_edge: Optional[pattern_graph.EdgeID],
    ) -> Optional[Dict[pattern_graph.EdgeID, List[Tuple[int, int, int]]]]:
        edge_id_to_data_choices = {}

        candidate = intermediate.node_ids_to_data_ids[nid]
        for neighbor_id in self.pgraph._node_out_incident_edges.get(nid, []):
            if neighbor_id == extended_edge:
                continue
            neighbor = self.pgraph.edges[neighbor_id]
            other_n = neighbor.end
            if other_n != nid and other_n not in intermediate.node_ids_to_data_ids:
                continue
            data_id = intermediate.node_ids_to_data_ids[other_n]
            if edges := self.find_all_edges(candidate, data_id, neighbor):
                edge_id_to_data_choices[neighbor_id] = edges
            else:
                return None

        for neighbor_id in self.pgraph._node_in_incident_edges.get(nid, []):
            if neighbor_id == extended_edge:
                continue
            neighbor = self.pgraph.edges[neighbor_id]
            other_n = neighbor.start
            if other_n != nid and other_n not in intermediate.node_ids_to_data_ids:
                continue
            data_id = intermediate.node_ids_to_data_ids[other_n]
            if edges := self.find_all_edges(data_id, candidate, neighbor):
                edge_id_to_data_choices[neighbor_id] = edges
            else:
                return None

        for neighbor_id in self.pgraph._node_undir_incident_edges.get(nid, []):
            if neighbor_id == extended_edge:
                continue
            neighbor = self.pgraph.edges[neighbor_id]
            other_n = neighbor.end if neighbor.start == nid else neighbor.start
            if other_n != nid and other_n not in intermediate.node_ids_to_data_ids:
                continue
            data_id = intermediate.node_ids_to_data_ids[other_n]
            edges = []
            if out_edges := self.find_all_edges(candidate, data_id, neighbor):
                edges += out_edges
            if in_edges := self.find_all_edges(data_id, candidate, neighbor):
                edges += in_edges
            if len(edges):
                edge_id_to_data_choices[neighbor_id] = list(set(edges))
            else:
                return None

        return edge_id_to_data_choices

    def find_nodes_connected_to(
        self,
        source: int,
        pedge: pattern_graph.Edge,
        pnode: pattern_graph.Node,
        check_out: bool,
        check_in: bool,
    ) -> List[Tuple[Tuple[int, int, int], int]]:
        results = []
        if check_out:
            # outgoing to self, in-coming from source
            for edge in self.graph.in_edges(source, keys=True):
                if self.edge_matches(pedge, edge) and self.node_matches(pnode, edge[0]):
                    results.append((edge, edge[0]))
        if check_in:
            # incoming to self, out-going from source
            for edge in self.graph.out_edges(source, keys=True):
                if check_out and edge[1] == source:
                    # self-loop is already matched as outgoing to self
                    continue
                if self.edge_matches(pedge, edge) and self.node_matches(pnode, edge[1]):
                    results.append((edge, edge[1]))
        return results

    def match_dfs(self, initial_matched: MatchResult) -> MatchResultSet:
        for node_id, data_id in initial_matched.node_ids_to_data_ids.items():
            if not self.node_matches(self.pgraph.nodes[node_id], data_id):
                return MatchResultSet()
        for edge_id, data_id in initial_matched.edge_ids_to_data_ids.items():
            if not self.edge_matches(self.pgraph.edges[edge_id], data_id):
                return MatchResultSet()

        num_matched = len(initial_matched.node_ids_to_data_ids)
        iteration_order = [id_ for id_ in initial_matched.node_ids_to_data_ids]
        while len(iteration_order) != len(self.pgraph.nodes):
            connected_node: Optional[pattern_graph.NodeID] = None
            degree: Optional[int] = None

            def get_min_degree_neighbor(
                node: pattern_graph.NodeID, neighbors: List[pattern_graph.EdgeID]
            ):
                connected_node: Optional[pattern_graph.NodeID] = None
                degree: Optional[int] = None
                for neighbor_id in neighbors:
                    neighbor = self.pgraph.edges[neighbor_id]
                    other_n = neighbor.end if neighbor.start == node else neighbor.start
                    if other_n in iteration_order:
                        continue
                    curr_deg = self.pgraph.degree(other_n)
                    if degree is None or curr_deg < degree:
                        degree = curr_deg
                        connected_node = other_n
                return connected_node, degree

            for node in iteration_order:
                curr_node, curr_deg = get_min_degree_neighbor(
                    node, self.pgraph.out_neighbors(node)
                )
                if curr_deg:
                    if degree is None or curr_deg < degree:
                        degree = curr_deg
                        connected_node = curr_node

                curr_node, curr_deg = get_min_degree_neighbor(
                    node, self.pgraph.in_neighbors(node)
                )
                if curr_deg:
                    if degree is None or curr_deg < degree:
                        degree = curr_deg
                        connected_node = curr_node

                curr_node, curr_deg = get_min_degree_neighbor(
                    node, self.pgraph.undir_neighbors(node)
                )
                if curr_deg:
                    if degree is None or curr_deg < degree:
                        degree = curr_deg
                        connected_node = curr_node

            if not connected_node:
                # pick the node with the lowest degree
                for curr_node in self.pgraph.nodes:
                    if curr_node in iteration_order:
                        continue
                    curr_deg = self.pgraph.degree(curr_node)
                    if degree is None or curr_deg < degree:
                        degree = curr_deg
                        connected_node = curr_node
                assert connected_node
            iteration_order.append(connected_node)
        self._match_dfs(iteration_order[num_matched:], initial_matched)
        return self.results

    def _combine_edges(
        self,
        intermediate: MatchResult,
        edges: Dict[pattern_graph.EdgeID, List[Tuple[int, int, int]]],
        cb,
    ):
        if len(edges) == 0:
            return cb(intermediate)

        k = list(edges.keys())[0]
        edges_to_bind = edges[k]
        del edges[k]
        for edge in edges_to_bind:
            if intermediate.contains_edge(edge):
                continue
            intermediate.edge_ids_to_data_ids[k] = edge
            self._combine_edges(intermediate, edges, cb)

    def _match_dfs(
        self, iteration_order: List[pattern_graph.NodeID], intermediate: MatchResult
    ):
        if len(iteration_order) == 0:
            self.results.add(intermediate)
            return

        nid = iteration_order[0]
        n = self.pgraph.nodes[nid]

        found = None
        picked_neighbor = None
        # Attempt to see if a neighbor is already matched
        for neighbor_id in self.pgraph._node_out_incident_edges.get(nid, []):
            neighbor = self.pgraph.edges[neighbor_id]
            other_n = neighbor.end
            if other_n not in intermediate.node_ids_to_data_ids:
                continue
            picked_neighbor = neighbor_id
            data_id = intermediate.node_ids_to_data_ids[other_n]
            found = self.find_nodes_connected_to(data_id, neighbor, n, True, False)

        if picked_neighbor is None:
            for neighbor_id in self.pgraph._node_in_incident_edges.get(nid, []):
                neighbor = self.pgraph.edges[neighbor_id]
                other_n = neighbor.start
                if other_n not in intermediate.node_ids_to_data_ids:
                    continue
                picked_neighbor = neighbor_id
                data_id = intermediate.node_ids_to_data_ids[other_n]
                found = self.find_nodes_connected_to(data_id, neighbor, n, False, True)

        if picked_neighbor is None:
            for neighbor_id in self.pgraph._node_undir_incident_edges.get(nid, []):
                neighbor = self.pgraph.edges[neighbor_id]
                other_n = neighbor.end if neighbor.start == nid else neighbor.start
                if other_n not in intermediate.node_ids_to_data_ids:
                    continue
                picked_neighbor = neighbor_id
                data_id = intermediate.node_ids_to_data_ids[other_n]
                found = self.find_nodes_connected_to(data_id, neighbor, n, True, True)

        if picked_neighbor is not None:
            assert found is not None
            for data_edge, data_node in found:
                if intermediate.contains_edge(data_edge):
                    continue

                tmp = intermediate.copy()
                tmp.node_ids_to_data_ids[nid] = data_node
                tmp.edge_ids_to_data_ids[picked_neighbor] = data_edge
                matched_edges = self.satisfies_edges(tmp, nid, picked_neighbor)
                if matched_edges is None:
                    continue
                self._combine_edges(
                    tmp,
                    matched_edges,
                    lambda x: self._match_dfs(iteration_order[1:], x),
                )
        else:
            # no neighbor is already matched, scan the whole graph
            for node in self.graph.nodes:
                if self.node_matches(n, node):
                    tmp = intermediate.copy()
                    tmp.node_ids_to_data_ids[nid] = node
                    matched_edges = self.satisfies_edges(tmp, nid, picked_neighbor)
                    if matched_edges is None:
                        continue
                    self._combine_edges(
                        tmp,
                        matched_edges,
                        lambda x: self._match_dfs(iteration_order[1:], x),
                    )
