"""Regression test for the ``nodes()`` built-in function.

The ``Graph`` abstract base is generic over the backend's edge type: backends
are free to use any Python value as an edge identifier and only have to expose
the endpoints through :meth:`Graph.src` and :meth:`Graph.dst`. Earlier
``spycy.functions.list_fns.nodes`` indexed each edge with ``edge[0]`` / ``edge[1]``
which accidentally hard-coded the assumption that edge IDs are subscriptable
``(src, dst, key)`` tuples. A backend using plain integer edge IDs would crash
with ``TypeError: 'int' object is not subscriptable`` on any query that calls
``nodes(p)`` over a variable-length path.

This test exercises that path with a minimal backend whose edge IDs are plain
``int`` s, so it fails on the old code and passes on the fix.

The test is written as a runnable script so it works without introducing a
test runner dependency. Invoke directly:

    python3 test/test_nodes_non_tuple_edges.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from spycy.graph import Graph
from spycy.spycy import CypherExecutorBase
from spycy.types import Node


NodeId = int
EdgeId = int  # deliberately non-subscriptable, unlike NetworkXGraph's tuple


@dataclass
class IntEdgeGraph(Graph[NodeId, EdgeId]):
    """Minimal in-memory Graph backend with plain integer edge IDs."""

    _nodes: Dict[NodeId, Dict[str, Any]] = field(default_factory=dict)
    _edges: Dict[EdgeId, Dict[str, Any]] = field(default_factory=dict)
    _endpoints: Dict[EdgeId, Tuple[NodeId, NodeId]] = field(default_factory=dict)
    _out: Dict[NodeId, List[EdgeId]] = field(default_factory=dict)
    _in: Dict[NodeId, List[EdgeId]] = field(default_factory=dict)
    _next_node_id: int = 0
    _next_edge_id: int = 0

    @property
    def nodes(self) -> Mapping[NodeId, Any]:
        return self._nodes

    @property
    def edges(self) -> Mapping[EdgeId, Any]:
        return self._edges

    def add_node(self, data: Dict[str, Any]) -> NodeId:
        nid = self._next_node_id
        self._next_node_id += 1
        self._nodes[nid] = data
        self._out[nid] = []
        self._in[nid] = []
        return nid

    def add_edge(
        self, start: NodeId, end: NodeId, data: Dict[str, Any]
    ) -> EdgeId:
        eid = self._next_edge_id
        self._next_edge_id += 1
        self._edges[eid] = data
        self._endpoints[eid] = (start, end)
        self._out[start].append(eid)
        self._in[end].append(eid)
        return eid

    def out_edges(self, node: NodeId) -> List[EdgeId]:
        return list(self._out.get(node, []))

    def in_edges(self, node: NodeId) -> List[EdgeId]:
        return list(self._in.get(node, []))

    def remove_node(self, node: NodeId) -> None:
        del self._nodes[node]
        self._out.pop(node, None)
        self._in.pop(node, None)

    def remove_edge(self, edge: EdgeId) -> None:
        del self._edges[edge]
        src, dst = self._endpoints.pop(edge)
        self._out[src].remove(edge)
        self._in[dst].remove(edge)

    def src(self, edge: EdgeId) -> NodeId:
        return self._endpoints[edge][0]

    def dst(self, edge: EdgeId) -> NodeId:
        return self._endpoints[edge][1]


class IntEdgeExecutor(CypherExecutorBase[NodeId, EdgeId]):
    def __init__(self) -> None:
        super().__init__(graph=IntEdgeGraph())


def _build_chain(exe: IntEdgeExecutor, length: int) -> List[NodeId]:
    """Create a linear chain ``n0 -> n1 -> ... -> n{length}`` and return the
    node IDs in order."""
    node_ids: List[NodeId] = []
    for i in range(length + 1):
        nid = exe.graph.add_node(
            {"labels": ["N"], "properties": {"idx": i}}
        )
        node_ids.append(nid)
    for a, b in zip(node_ids, node_ids[1:]):
        exe.graph.add_edge(a, b, {"type": "R", "properties": {}})
    return node_ids


def test_nodes_fn_returns_chain_on_non_tuple_edge_backend() -> None:
    exe = IntEdgeExecutor()
    chain = _build_chain(exe, length=3)

    result = exe.exec(
        "MATCH p=(a)-[*1..]->(b) WHERE a.idx = 0 AND b.idx = 3 RETURN nodes(p) AS ns"
    )

    assert len(result) == 1, f"expected exactly one full-chain match, got {len(result)}"
    ns = result["ns"].iloc[0]
    assert isinstance(ns, list), f"nodes() must return a list, got {type(ns)}"
    assert [n.id_ for n in ns] == chain, (
        f"expected nodes() to yield the chain {chain}, got {[n.id_ for n in ns]}"
    )


def test_nodes_fn_on_single_edge_path() -> None:
    exe = IntEdgeExecutor()
    chain = _build_chain(exe, length=1)

    result = exe.exec(
        "MATCH p=(a)-->(b) WHERE a.idx = 0 AND b.idx = 1 RETURN nodes(p) AS ns"
    )

    assert len(result) == 1
    ns = result["ns"].iloc[0]
    assert [n.id_ for n in ns] == chain


def _run_all() -> None:
    test_nodes_fn_returns_chain_on_non_tuple_edge_backend()
    test_nodes_fn_on_single_edge_path()
    print("OK: nodes() regression tests passed")


if __name__ == "__main__":
    _run_all()
