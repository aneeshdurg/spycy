# sPyCy

sPyCy is a python implementation of [openCypher](https://github.com/opencypher/openCypher/).
Try it out in your browser here: [https://aneeshdurg.me/spycy](https://aneeshdurg.me/spycy)

The goal of `sPyCy` is to have a simple in-memory graph database engine that is
not concerned with performance. The ideal use-case is for testing programs that
generate small openCypher queries, or as an alternative reference
implementation.

The long-term goals of this project include fully passing all openCypher TCK
tests.

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

spycy is still mostly unimplemented so most queries will probably error or
return incorrect results.
