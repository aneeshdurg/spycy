# sPyCy

![TCK CI status](https://github.com/aneeshdurg/spycy/actions/workflows/tck.yml/badge.svg)

sPyCy is a python implementation of [openCypher](https://github.com/opencypher/openCypher/).
Try it out in your browser here: [https://aneeshdurg.me/spycy](https://aneeshdurg.me/spycy)

The goal of `sPyCy` is to have a simple in-memory graph database engine that is
not concerned with performance. The ideal use-case is for testing programs that
generate small openCypher queries, or as an alternative reference
implementation.

The long-term goals of this project include fully passing all openCypher TCK
tests. Currently a majority of TCK tests are expected to pass. The failing tests
are documented.

Some major **unimplemented** features are `MERGE`, temporal values, `WHERE`
predicates involving patterns, existential subqueries, and `CALL`.

## Installation

You can either install from the pre-built wheel or build `sPyCy` yourself. To install the pre-built wheel, run:

```bash
pip install https://aneeshdurg.me/spycy/dist/spycy_aneeshdurg-0.0.1-py3-none-any.whl
```

To build it yourself, from the root of this repo, run:

```bash
python3 -m build
```

## Usage:

```bash
# Example usage:
python3 -m spycy --query "CREATE (a {id: 'node0'})-[:X]->(b) return a"

# Or interactive mode:
python3 -m spycy --interactive
> CREATE (a {id: 'node0'})-[:X]->(b {id: 'node1'})
> MATCH (a)--(b) RETURN a.id, b.id
    a.id   b.id
0  node0  node1
```

`sPcY` can also be used via python:
```python
from spycy import spycy
exe = spycy.CypherExecutor()
ids = list(range(100))
exe.exec(f"UNWIND {ids} as x CREATE ({{id: x}})")
num_nodes_gt_10 = exe.exec("MATCH (a) WHERE a.id > 10 RETURN count(a) as output")["output"]
print(f"There are {num_nodes_gt_10} node(s) with an id greater than 10")
```
