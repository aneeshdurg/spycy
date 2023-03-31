from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from gen.CypherParser import CypherParser


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
    start: Optional[int]
    end: Optional[int]


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
class Graph:
    nodes: Dict[NodeID, Node] = field(default_factory=dict)
    edges: Dict[EdgeID, Edge] = field(default_factory=dict)
    _node_name_to_id: Dict[str, NodeID] = field(default_factory=dict)
    _edge_name_to_id: Dict[str, EdgeID] = field(default_factory=dict)

    def add_relationship(
        self,
        rel_el: CypherParser.OC_RelationshipPatternContext,
        start: NodeID,
        end: NodeID,
    ):
        incoming = bool(rel_el.oC_LeftArrowHead())
        outgoing = bool(rel_el.oC_RightArrowHead())
        undirected = (incoming and outgoing) or not (incoming or outgoing)
        if undirected:
            incoming = False
            outgoing = False
        if incoming:
            start, end = end, start

        name = None
        edge_range = None
        details = rel_el.oC_RelationshipDetail()
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
                edge_range = EdgeRange(None, None)
                for child in range_lit.children:
                    if isinstance(child, CypherParser.OC_IntegerLiteralContext):
                        value = eval(child.getText())
                        if has_intermediate:
                            edge_range.end = value
                        else:
                            edge_range.start = value
                    elif child.getText() == "..":
                        has_intermediate = True

            if props := details.oC_Properties():
                assert not props.oC_Parameter(), "Unsupported query - parameters"
                if map_lit := props.oC_MapLiteral():
                    properties = map_lit

        edgeid = EdgeID(len(self.edges))
        edge = Edge(edgeid, name, undirected, start, end, edge_range, types, properties)
        self.edges[edgeid] = edge
        if name:
            assert name not in self._edge_name_to_id
            self._edge_name_to_id[name] = edgeid

    def add_node(self, node_el: CypherParser.OC_NodePatternContext) -> NodeID:
        name_var = node_el.oC_Variable()
        name = None
        if name_var:
            name = name_var.getText()
        nodeid = None
        if name:
            nodeid = self._node_name_to_id.get(name)
        if not nodeid:
            nodeid = NodeID(len(self.nodes))
            self.nodes[nodeid] = Node(nodeid, name)
            if name:
                self._node_name_to_id[name] = nodeid

        node = self.nodes[nodeid]

        if label_el := node_el.oC_NodeLabels():
            labels = label_el.oC_NodeLabel()
            assert labels
            for label in labels:
                lname = label.oC_LabelName()
                assert lname
                node.add_label(lname.getText())

        if props := node_el.oC_Properties():
            assert not props.oC_Parameter(), "Unsupported query - parameters"
            if map_lit := props.oC_MapLiteral():
                node.properties = map_lit

        return nodeid

    def add_fragment(self, fragment: CypherParser.OC_AnonymousPatternPartContext):
        element = fragment.oC_PatternElement()
        assert element
        while inner_el := element.oC_PatternElement():
            element = inner_el

        node_el = element.oC_NodePattern()
        assert node_el
        start_node = self.add_node(node_el)

        chain = element.oC_PatternElementChain()
        if not chain:
            return
        for chain_el in chain:
            rel_el = chain_el.oC_RelationshipPattern()
            assert rel_el
            endpt_el = chain_el.oC_NodePattern()
            assert endpt_el

            end_node = self.add_node(endpt_el)
            self.add_relationship(rel_el, start_node, end_node)
            start_node = end_node
