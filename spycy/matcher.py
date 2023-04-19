from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

import spycy.pattern_graph as pattern_graph
from spycy.graph import DataEdge, Graph

# Either a single edge, or a variable length relationship
MatchedEdge = Union[DataEdge, List[DataEdge]]


@dataclass
class MatchResult:
    node_ids_to_data_ids: Dict[pattern_graph.NodeID, int] = field(default_factory=dict)
    edge_ids_to_data_ids: Dict[pattern_graph.EdgeID, MatchedEdge] = field(
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
class MatchResultSet:
    node_ids_to_data_ids: Dict[pattern_graph.NodeID, List[int]] = field(
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


class Matcher(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def match(
        cls,
        graph: Graph,
        pgraph: pattern_graph.Graph,
        row_id: int,
        node_ids_to_props: Dict[pattern_graph.NodeID, pd.Series],
        edge_ids_to_props: Dict[pattern_graph.EdgeID, pd.Series],
        initial_matched: MatchResult,
    ) -> MatchResultSet:
        pass


@dataclass
class DFSMatcher(Matcher):
    graph: Graph
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

    def edge_matches(self, pedge: pattern_graph.Edge, data_edge: DataEdge) -> bool:
        """Check if `data_edge` matches `pedge`, ignoring variable length attributes"""
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
    ) -> List[MatchedEdge]:
        """Find all edges from source to dst (directed only, doesn't handle
        variable length undirected edges)"""
        if pedge.range_:
            assert not pedge.undirected
            extension_node = pattern_graph.Node(pattern_graph.NodeID(-1), None)
            return [
                e[0]
                for e in self.variable_length_relationship(
                    source, pedge, extension_node, False, True, dst
                )
            ]
        output = []
        for edge in self.graph.out_edges(source):
            if edge[1] != dst:
                continue
            if self.edge_matches(pedge, edge):
                output.append(edge)
        return output

    def satisfies_edges(
        self,
        intermediate: MatchResult,
        nid: pattern_graph.NodeID,
        extended_edge: Optional[pattern_graph.EdgeID],
    ) -> Optional[Dict[pattern_graph.EdgeID, List[DataEdge]]]:
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
            if neighbor.range_ and neighbor.undirected:
                extension_node = pattern_graph.Node(pattern_graph.NodeID(-1), None)
                edges = [
                    e[0]
                    for e in self.variable_length_relationship(
                        candidate, neighbor, extension_node, True, True, data_id
                    )
                ]
            else:
                if out_edges := self.find_all_edges(candidate, data_id, neighbor):
                    edges += out_edges
                if in_edges := self.find_all_edges(data_id, candidate, neighbor):
                    edges += in_edges
            if len(edges):
                edge_id_to_data_choices[neighbor_id] = list(set(edges))
            else:
                return None

        return edge_id_to_data_choices

    def variable_length_relationship(
        self,
        source: int,
        pedge: pattern_graph.Edge,
        pnode: pattern_graph.Node,
        check_out: bool,
        check_in: bool,
        dst: Optional[int] = None,
    ) -> List[Tuple[MatchedEdge, int]]:
        assert pedge.range_

        output = []
        start = pedge.range_.start
        end = pedge.range_.end
        # pattern Node with no properties or labels to allow matching any data node
        extension_node = pattern_graph.Node(pattern_graph.NodeID(-1), None)

        def variable_length_dfs(depth: int, state: Tuple[List[DataEdge], int]):
            if depth >= start:
                if dst is not None:
                    if state[1] == dst:
                        output.append(state)
                elif self.node_matches(pnode, state[1]):
                    output.append(state)
            if depth < end:
                # Do extension, but have no restriction on the node
                next_edges = self.find_nodes_connected_to(
                    state[1],
                    pedge,
                    extension_node,
                    check_out,
                    check_in,
                    ignore_var_length=True,
                )
                for (next_edge, next_node) in next_edges:
                    assert not isinstance(next_edge, list)
                    found_conflict = False
                    for e in state[0]:
                        if e == next_edge:
                            found_conflict = True
                            break
                    if found_conflict:
                        continue
                    next_state = (state[0] + [next_edge], next_node)
                    variable_length_dfs(depth + 1, next_state)
                pass

        variable_length_dfs(0, ([], source))
        return output

    def find_nodes_connected_to(
        self,
        source: int,
        pedge: pattern_graph.Edge,
        pnode: pattern_graph.Node,
        check_out: bool,
        check_in: bool,
        ignore_var_length: bool = False,
    ) -> List[Tuple[MatchedEdge, int]]:
        results = []
        if pedge.range_ and not ignore_var_length:
            return self.variable_length_relationship(
                source, pedge, pnode, check_out, check_in
            )
        if check_out:
            # outgoing to self, in-coming from source
            for edge in self.graph.in_edges(source):
                if self.edge_matches(pedge, edge) and self.node_matches(pnode, edge[0]):
                    results.append((edge, edge[0]))
        if check_in:
            # incoming to self, out-going from source
            for edge in self.graph.out_edges(source):
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
            pedge = self.pgraph.edges[edge_id]
            if not self.edge_matches(pedge, data_id):
                return MatchResultSet()

            def check_node(pnode: pattern_graph.NodeID, dnode: int) -> bool:
                if init_n := initial_matched.node_ids_to_data_ids.get(pnode):
                    if init_n != dnode:
                        return False
                else:
                    if not self.node_matches(self.pgraph.nodes[pnode], dnode):
                        return False
                return True

            src, dst, _ = data_id
            src_pnode = pedge.start
            if not check_node(src_pnode, src):
                return MatchResultSet()
            if src_pnode in initial_matched.node_ids_to_data_ids:
                if initial_matched.node_ids_to_data_ids[src_pnode] != src:
                    return MatchResultSet()
            else:
                initial_matched.node_ids_to_data_ids[src_pnode] = src
            dst_pnode = pedge.end
            if not check_node(dst_pnode, dst):
                return MatchResultSet()
            if dst_pnode in initial_matched.node_ids_to_data_ids:
                if initial_matched.node_ids_to_data_ids[dst_pnode] != dst:
                    return MatchResultSet()
            else:
                initial_matched.node_ids_to_data_ids[dst_pnode] = dst
            if pedge.undirected:
                raise AssertionError("Not implemented binding as undirected")

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
        edges: Dict[pattern_graph.EdgeID, List[DataEdge]],
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

    @classmethod
    def match(
        cls,
        graph: Graph,
        pgraph: pattern_graph.Graph,
        row_id: int,
        node_ids_to_props: Dict[pattern_graph.NodeID, pd.Series],
        edge_ids_to_props: Dict[pattern_graph.EdgeID, pd.Series],
        initial_matched: MatchResult,
    ) -> MatchResultSet:
        matcher = DFSMatcher(
            graph, pgraph, row_id, node_ids_to_props, edge_ids_to_props
        )
        return matcher.match_dfs(initial_matched)
