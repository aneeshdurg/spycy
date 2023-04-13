import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from spycy.gen.CypherParser import CypherParser


@dataclass
class NodeID:
    _id: int

    def __hash__(self):
        return self._id


@dataclass
class EdgeID:
    _id: int

    def __hash__(self):
        return self._id


@dataclass
class Node:
    id_: NodeID
    name: Optional[str]

    labels: Set[str] = field(default_factory=set)
    properties: Optional[CypherParser.OC_MapLiteralContext] = None

    def add_label(self, label: str):
        self.labels.add(label)


@dataclass
class EdgeRange:
    start: float
    end: float


@dataclass
class Edge:
    id_: EdgeID
    name: Optional[str]
    undirected: bool
    start: NodeID
    end: NodeID
    range_: Optional[EdgeRange]
    types: Set[str]
    properties: Optional[CypherParser.OC_MapLiteralContext]

    def add_type(self, type_: str):
        self.types.add(type_)


@dataclass
class Path:
    nodes: List[NodeID]
    edges: List[EdgeID]


@dataclass
class Graph:
    nodes: Dict[NodeID, Node] = field(default_factory=dict)
    edges: Dict[EdgeID, Edge] = field(default_factory=dict)
    paths: Dict[str, Path] = field(default_factory=dict)

    _node_name_to_id: Dict[str, NodeID] = field(default_factory=dict)
    _edge_name_to_id: Dict[str, EdgeID] = field(default_factory=dict)

    _node_out_incident_edges: Dict[NodeID, List[EdgeID]] = field(default_factory=dict)
    _node_in_incident_edges: Dict[NodeID, List[EdgeID]] = field(default_factory=dict)
    _node_undir_incident_edges: Dict[NodeID, List[EdgeID]] = field(default_factory=dict)

    def out_neighbors(self, node: NodeID) -> List[EdgeID]:
        return self._node_out_incident_edges.get(node, [])

    def in_neighbors(self, node: NodeID) -> List[EdgeID]:
        return self._node_in_incident_edges.get(node, [])

    def undir_neighbors(self, node: NodeID) -> List[EdgeID]:
        return self._node_undir_incident_edges.get(node, [])

    def out_degree(self, node: NodeID) -> int:
        return len(self.out_neighbors(node))

    def in_degree(self, node: NodeID) -> int:
        return len(self.in_neighbors(node))

    def undir_degree(self, node: NodeID) -> int:
        return len(self.undir_neighbors(node))

    def degree(self, node: NodeID) -> int:
        return self.out_degree(node) + self.in_degree(node) + self.undir_degree(node)

    @classmethod
    def build_edge(
        cls,
        edgeid: EdgeID,
        undirected: bool,
        start: NodeID,
        end: NodeID,
        details: Optional[CypherParser.OC_RelationshipDetailContext],
    ) -> Edge:
        name = None
        edge_range = None
        types = set()
        properties = None
        if details:
            if name_el := details.oC_Variable():
                name = name_el.getText()

            if rel_types := details.oC_RelationshipTypes():
                type_names = rel_types.oC_RelTypeName()
                assert type_names
                for type_name in type_names:
                    types.add(type_name.getText())

            if range_lit := details.oC_RangeLiteral():
                assert range_lit.children
                has_intermediate = False
                range_: List[Optional[float]] = [None, None]
                for child in range_lit.children:
                    if isinstance(child, CypherParser.OC_IntegerLiteralContext):
                        value = eval(child.getText())
                        if has_intermediate:
                            range_[1] = value
                        else:
                            range_[0] = value
                    elif child.getText() == "..":
                        has_intermediate = True

                if not has_intermediate:
                    if range_[0] is None:
                        range_[0] = 1
                        range_[1] = math.inf
                    else:
                        range_[1] = range_[0]
                else:
                    if range_[0] is None:
                        range_[0] = 1

                    if range_[1] is None:
                        range_[1] = math.inf

                edge_range = EdgeRange(range_[0], range_[1])

            if props := details.oC_Properties():
                assert not props.oC_Parameter(), "Unsupported query - parameters"
                if map_lit := props.oC_MapLiteral():
                    properties = map_lit

        return Edge(edgeid, name, undirected, start, end, edge_range, types, properties)

    def add_relationship(
        self,
        rel_el: CypherParser.OC_RelationshipPatternContext,
        start: NodeID,
        end: NodeID,
    ) -> EdgeID:
        incoming = bool(rel_el.oC_LeftArrowHead())
        outgoing = bool(rel_el.oC_RightArrowHead())
        undirected = (incoming and outgoing) or not (incoming or outgoing)
        if undirected:
            incoming = False
            outgoing = False
        if incoming:
            start, end = end, start

        details = rel_el.oC_RelationshipDetail()

        edgeid = EdgeID(len(self.edges))
        edge = Graph.build_edge(edgeid, undirected, start, end, details)
        self.edges[edgeid] = edge
        if edge.name:
            assert edge.name not in self._edge_name_to_id
            self._edge_name_to_id[edge.name] = edgeid

        if undirected:
            if start not in self._node_undir_incident_edges:
                self._node_undir_incident_edges[start] = []
            if end not in self._node_undir_incident_edges:
                self._node_undir_incident_edges[end] = []
            self._node_undir_incident_edges[start].append(edgeid)
            self._node_undir_incident_edges[end].append(edgeid)
        else:
            if start not in self._node_out_incident_edges:
                self._node_out_incident_edges[start] = []
            self._node_out_incident_edges[start].append(edgeid)

            if end not in self._node_in_incident_edges:
                self._node_in_incident_edges[end] = []
            self._node_in_incident_edges[end].append(edgeid)
        return edgeid

    def add_node(self, node_el: CypherParser.OC_NodePatternContext) -> NodeID:
        name_var = node_el.oC_Variable()
        name = None
        if name_var:
            name = name_var.getText()
        nodeid = None
        preexisting = False
        if name:
            nodeid = self._node_name_to_id.get(name)
            if nodeid:
                preexisting = True
        if not nodeid:
            nodeid = NodeID(len(self.nodes))
            self.nodes[nodeid] = Node(nodeid, name)
            if name:
                self._node_name_to_id[name] = nodeid

        node = self.nodes[nodeid]

        if label_el := node_el.oC_NodeLabels():
            if preexisting:
                raise Exception("SyntaxError::VariableAlreadyBound")
            labels = label_el.oC_NodeLabel()
            assert labels
            for label in labels:
                lname = label.oC_LabelName()
                assert lname
                node.add_label(lname.getText())

        if props := node_el.oC_Properties():
            if preexisting:
                raise Exception("SyntaxError::VariableAlreadyBound")
            assert not props.oC_Parameter(), "Unsupported query - parameters"
            if map_lit := props.oC_MapLiteral():
                node.properties = map_lit

        return nodeid

    def add_fragment(
        self,
        fragment: CypherParser.OC_AnonymousPatternPartContext,
        path_name: Optional[str],
    ):
        created_nodes = []
        created_edges = []

        element = fragment.oC_PatternElement()
        assert element
        while inner_el := element.oC_PatternElement():
            element = inner_el

        node_el = element.oC_NodePattern()
        assert node_el
        start_node = self.add_node(node_el)

        created_nodes.append(start_node)

        chain = element.oC_PatternElementChain()
        if chain is None:
            chain = []

        for chain_el in chain:
            rel_el = chain_el.oC_RelationshipPattern()
            assert rel_el
            endpt_el = chain_el.oC_NodePattern()
            assert endpt_el

            end_node = self.add_node(endpt_el)
            created_nodes.append(end_node)

            edge = self.add_relationship(rel_el, start_node, end_node)
            start_node = end_node

            created_edges.append(edge)

        if path_name:
            self.paths[path_name] = Path(created_nodes, created_edges)
