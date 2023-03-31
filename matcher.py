from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

import pattern_graph


@dataclass
class MatchResult:
    node_ids_to_data_ids: Dict[pattern_graph.NodeID, int] = field(default_factory=dict)
    edge_ids_to_data_ids: Dict[pattern_graph.EdgeID, Tuple[int, int, int]] = field(
        default_factory=dict
    )

    def copy(self) -> "MatchResult":
        return MatchResult(
            self.node_ids_to_data_ids.copy(), self.edge_ids_to_data_ids.copy()
        )


@dataclass
class Matcher:
    graph: nx.MultiDiGraph
    pgraph: pattern_graph.Graph

    def match_dfs(self) -> List[MatchResult]:
        iteration_order = []
        while len(iteration_order) != len(self.pgraph.nodes):
            connected_node: Optional[pattern_graph.NodeID] = None
            degree: Optional[int] = None

            def get_min_degree_neighbor(
                node: pattern_graph.NodeID, neighbors: List[pattern_graph.EdgeID]
            ):
                connected_node: Optional[pattern_graph.NodeID] = None
                degree: Optional[int] = None
                for neighbor in neighbors:
                    other_n = neighbor.end if neighbor.start == node else neighbor.start
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
        return_list = []
        self._match_dfs(iteration_order, MatchResult(), return_list)
        return return_list

    def _match_dfs(
        self,
        iteration_order: List[pattern_graph.NodeID],
        intermediate: MatchResult,
        results: List[MatchResult],
    ):
        if len(iteration_order) == 0:
            results.append(intermediate.copy())
            return

        nid = iteration_order[0]
        n = self.pgraph.nodes[nid]

        found = False
        # Attempt to see if a neighbor is already matched
        for neighbor in self.pgraph._node_out_incident_edges.get(nid, []):
            if neighbor not in intermediate.node_ids_to_data_ids:
                continue
            data_id = intermediate.node_ids_to_data_ids[neighbor]
            for data_edge in self.graph.edges(data_id):
                print("O", nid, neighbor, data_id, data_edge)

        for neighbor in self.pgraph._node_in_incident_edges.get(nid, []):
            if neighbor not in intermediate.node_ids_to_data_ids:
                continue
            data_id = intermediate.node_ids_to_data_ids[neighbor]
            for data_edge in self.graph.in_edges(data_id):
                print("I", nid, neighbor, data_id, data_edge)

        for neighbor in self.pgraph._node_undir_incident_edges.get(nid, []):
            if neighbor not in intermediate.node_ids_to_data_ids:
                continue
            data_id = intermediate.node_ids_to_data_ids[neighbor]
            for data_edge in self.graph.edges(data_id):
                print("U", nid, neighbor, data_id, data_edge)
            for data_edge in self.graph.in_edges(data_id):
                print("U", nid, neighbor, data_id, data_edge)

        if not found:
            # no neighbor is already matched, scan the whole graph
            for node in self.graph.nodes:
                if not n.labels and not n.properties:
                    intermediate.node_ids_to_data_ids[nid] = node
                    self._match_dfs(iteration_order[1:], intermediate, results)
                else:
                    assert False
