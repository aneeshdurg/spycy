#!/usr/bin/env python3
"""sPyCy DOM - an example of using sPyCy to query DOM trees"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from bs4 import BeautifulSoup

from spycy.graph import Graph
from spycy.spycy import CypherExecutorBase

NodeType = int
EdgeType = Tuple[int, int]


@dataclass
class SoupGraph(Graph[NodeType, EdgeType]):
    """A Graph implementation that maps graph access to a HTML DOM"""

    soup: BeautifulSoup
    node_map: Dict[int, Any] = field(default_factory=dict)
    edge_map: Dict[Tuple[int, int], Any] = field(default_factory=dict)

    def __post_init__(self):
        self.node_map = {
            hash(s): {
                "labels": [s.name],
                "properties": {"attrs": s.attrs, "text": s.text},
                "obj": s,
            }
            for s in self.soup.find_all()
        }

        for n, node in self.node_map.items():
            for c in node["obj"].children:
                self.edge_map[(n, hash(c))] = {"type": "child", "properties": {}}

    @property
    def nodes(self) -> Mapping[NodeType, Any]:
        return self.node_map

    @property
    def edges(self) -> Mapping[EdgeType, Any]:
        return self.edge_map

    def add_node(self, *_) -> NodeType:
        raise Exception("SoupGraph is read-only!")

    def add_edge(self, *_) -> EdgeType:
        raise Exception("SoupGraph is read-only!")

    def out_edges(self, node: NodeType) -> List[EdgeType]:
        el = self.node_map[node]["obj"]
        return [(node, hash(child)) for child in el.children if child.name is not None]

    def in_edges(self, node: NodeType) -> List[EdgeType]:
        el = self.node_map[node]["obj"]
        return [(node, hash(el.parent))]

    def remove_node(self, _):
        raise Exception("SoupGraph is read-only!")

    def remove_edge(self, _):
        raise Exception("SoupGraph is read-only!")

    def src(self, edge: EdgeType) -> NodeType:
        return edge[0]

    def dst(self, edge: EdgeType) -> NodeType:
        return edge[1]


class SoupCypherExecutor(CypherExecutorBase[NodeType, EdgeType]):
    """Enable openCypher queries against a SoupGraph"""

    def __init__(self, input_: Path):
        """input_ should be a Path to an HTML document"""
        with input_.open() as f:
            soup = BeautifulSoup(f.read(), features="html5lib")
        soup_graph = SoupGraph(soup)
        super().__init__(graph=soup_graph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", action="store", type=Path)

    args = parser.parse_args()
    exe = SoupCypherExecutor(args.filename)
    print("Ordering elements by child count")
    print(
        exe.exec(
            "match (a)-->(b) return labels(a)[0] as tag, id(a), count(b) ORDER BY count(b)"
        )
    )
    print()

    print("Find all <p> elements")
    print(exe.exec("match (a:p) return a"))
    print()

    print("Find all <a> elements and extract their links")
    print(exe.exec("match (a:a) return a.attrs.href"))


if __name__ == "__main__":
    main()
