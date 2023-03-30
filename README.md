# PyPher

PyPher is a python implementation of
[openCypher](https://github.com/opencypher/openCypher/). The goal of `PyPher` is
to have a simple in-memory graph database engine that is not concerned with
performance. The ideal use-case is for testing programs that generate small
openCypher queries, or as an alternative reference implementation.

The long-term goals of this project include fully passing all openCypher TCK
tests.

## Currently supported:
+ Simple expressions on scalars (POD scalars only)

## TODO:
+ Add tests
+ Implement list/map support.
+ Implement support for `UNWIND`
+ Implement basic support for read only queries
