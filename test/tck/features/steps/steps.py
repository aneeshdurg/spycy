from typing import Any, Dict, List, Optional, Tuple

import hjson
import pandas as pd
from antlr4.error.ErrorListener import ErrorListener
from behave import given, then, when
from behave.model import Table

from antlr4 import *
from spycy import pattern_graph
from spycy.gen.CypherLexer import CypherLexer
from spycy.gen.CypherParser import CypherParser
from spycy.spycy import CypherExecutor
from spycy.types import Edge, Node

with open("openCypher/tck/graphs/binary-tree-1/binary-tree-1.cypher", "r") as f:
    BINARY_TREE1 = f.read()
with open("openCypher/tck/graphs/binary-tree-2/binary-tree-2.cypher", "r") as f:
    BINARY_TREE2 = f.read()


def execute_query(context: Any, query: str):
    context.result = None
    context.error = None
    try:
        context.result = context.executor.exec(query)
    except Exception as e:
        context.error = e


@given("an empty graph")
def empty_graph(context):
    pass


@given("any graph")
def any_graph(context):
    pass


@given("the binary-tree-1 graph")
def binary_tree1(context):
    raise Exception("Can't load graphs")


@given("the binary-tree-2 graph")
def binary_tree2(context):
    raise Exception("Can't load graphs")


@then("having executed")
@given("having executed")
@given("after having executed")
def setup_query(context):
    context.executor.exec(context.text)


@given("parameters are")
@given("parameter values are")
def define_params(context):
    print(context.table.headings)
    parameter_map = {}
    parameter_map[context.table.headings[0]] = context.table.headings[1]
    for row in context.table:
        parameter_map[row[0]] = row[1]
    context.executor.set_params(parameter_map)


@when("executing query")
@when("executing control query")
def when_query(context):
    execute_query(context, context.text)


def parse_tck_node(context, node_expr) -> Any:
    pgraph = context.executor._interpret_pattern(node_expr)
    assert len(pgraph.nodes) == 1
    node = list(pgraph.nodes.values())[0]
    if node.properties is None:
        node.properties = {}
    else:
        table = context.executor.table
        context.executor.reset_table()
        node.properties = context.executor._evaluate_map_literal(node.properties)[0]
        context.executor.table = table
    return node


def parse_tck_edge_details(context, edge_expr) -> Any:
    edge = pattern_graph.Graph.build_edge(
        pattern_graph.EdgeID(0),
        True,
        pattern_graph.NodeID(0),
        pattern_graph.NodeID(0),
        edge_expr,
    )
    if not edge.types:
        raise ValueError("Could not parse Edge")

    if edge.properties is None:
        edge.properties = {}
    else:
        table = context.executor.table
        context.executor.reset_table()
        edge.properties = context.executor._evaluate_map_literal(edge.properties)[0]
        context.executor.table = table
    return edge


def parse_tck_value(context, tck_expr) -> Any:
    if literal := tck_expr.tck_Literal():
        return hjson.loads(literal.getText())
    if pattern := tck_expr.oC_Pattern():
        print(type(tck_expr), type(pattern))
        return parse_tck_node(context, pattern)
    if edge := tck_expr.oC_RelationshipDetail():
        return parse_tck_edge_details(context, edge)
    if list_ := tck_expr.tck_List():
        return [parse_tck_value(context, v) for v in list_.tck_ExpectedValue()]
    if map_ := tck_expr.tck_Map():
        return {
            k.getText(): parse_tck_value(context, v)
            for k, v in zip(map_.oC_PropertyKeyName(), map_.tck_ExpectedValue())
        }

    raise Exception("Unexpected node type")


class TestErrorListener(ErrorListener):
    def syntaxError(self, *args):
        raise Exception("Parse error")


def normalize_tck_value(context, value: str) -> Any:
    error_listener = TestErrorListener()

    input_stream = InputStream(value)
    lexer = CypherLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    stream = CommonTokenStream(lexer)
    parser = CypherParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)

    root = parser.tck_ExpectedValue()
    return parse_tck_value(context, root)


def tck_to_records(context, table: Table) -> List[Dict[str, Any]]:
    records = []
    for row in table:
        row_obj = {}
        for heading in table.headings:
            row_obj[heading] = normalize_tck_value(context, row[heading])
        records.append(row_obj)
    return records


def normalize_spycy_value(context, value: Any) -> Any:
    if value is pd.NA:
        return None
    elif isinstance(value, list):
        return [normalize_spycy_value(context, v) for v in value]
    elif isinstance(value, dict):
        return {k: normalize_spycy_value(context, v) for k, v in value.items()}
    elif isinstance(value, Edge):
        id_ = pattern_graph.EdgeID(0)
        nid = pattern_graph.NodeID(0)
        data_edge = context.executor.graph.edges[value.id_]
        return pattern_graph.Edge(
            id_,
            None,
            True,
            nid,
            nid,
            None,
            set([data_edge["type"]]),
            data_edge["properties"],
        )
    elif isinstance(value, Node):
        data_node = context.executor.graph.nodes[value.id_]
        id_ = pattern_graph.NodeID(0)
        return pattern_graph.Node(
            id_, None, data_node["labels"], data_node["properties"]
        )
    return value


def normalize_spycy_output(
    context, py_table: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    for row in py_table:
        for k, v in row.items():
            row[k] = normalize_spycy_value(context, v)
    return py_table


def equivalent_ignoring_list_order(tck_value, spycy_value):
    if isinstance(tck_value, dict):
        if not isinstance(spycy_value, dict):
            return False
        if set(tck_value.keys()) != set(spycy_value.keys()):
            return False
        for k, tvalue in tck_value.items():
            if not equivalent_ignoring_list_order(tvalue, spycy_value[k]):
                return False
        return True

    if not isinstance(tck_value, list):
        return tck_value == spycy_value

    if not isinstance(spycy_value, list):
        return False
    if len(tck_value) != len(spycy_value):
        return False
    used_indicies = set()
    for svalue in spycy_value:
        found = False
        for i, tvalue in enumerate(tck_value):
            if i in used_indicies:
                continue
            if equivalent_ignoring_list_order(tvalue, svalue):
                used_indicies.add(i)
                found = True
                break
        if not found:
            return False
    return True


@then("the result should be (ignoring element order for lists)")
def assert_results_in_order_ignoring_list_order(context):
    if context.error:
        raise context.error
    print(context.result, context.table.headings)
    assert list(context.result.columns) == context.table.headings
    print(type(context.table))
    expected_rows = tck_to_records(context, context.table)
    assert len(context.result) == len(expected_rows)

    actual_rows = normalize_spycy_output(context, context.result.to_dict("records"))
    for expected, actual in zip(expected_rows, actual_rows):
        assert equivalent_ignoring_list_order(
            expected, actual
        ), f"{expected} != {actual}"


@then("the result should be, in any order")
def assert_results_in_any_order(context):
    if context.error:
        raise context.error
    print(context.result, context.table.headings)
    assert set(context.result.columns) == set(context.table.headings)
    expected_rows = tck_to_records(context, context.table)
    print(expected_rows)
    assert len(context.result) == len(expected_rows)

    actual_rows = normalize_spycy_output(context, context.result.to_dict("records"))
    used_rows = set()
    for actual in actual_rows:
        print("??", actual)
        found = False
        for i, expected in enumerate(expected_rows):
            print(" ???", expected)
            if i in used_rows:
                continue
            if expected == actual:
                used_rows.add(i)
                found = True
                break
        assert found, f"{actual} not in expected"


@then("the result should be, in order")
def assert_results_in_order(context):
    if context.error:
        raise context.error
    print(context.result, context.table.headings)
    assert list(context.result.columns) == context.table.headings
    print(type(context.table))
    expected_rows = tck_to_records(context, context.table)
    assert len(context.result) == len(expected_rows)

    actual_rows = normalize_spycy_output(context, context.result.to_dict("records"))
    for expected, actual in zip(expected_rows, actual_rows):
        assert expected == actual, f"{expected} != {actual}"


@then("the result should be empty")
def assert_empty_result(context):
    if context.error:
        raise context.error
    assert len(context.result) == 0, len(context.result)


@then("{errorType} should be raised at runtime: {error}")
@then("a {errorType} should be raised at runtime: {error}")
@then("{errorType} should be raised at compile time: {error}")
@then("a {errorType} should be raised at compile time: {error}")
@then("a {errorType} should be raised at any time: {error}")
def assert_error(context, errorType, error):
    assert context.error is not None


@then("the side effects should be")
def assert_side_effects(context):
    pass


@then("no side effects")
def assert_no_side_effects(context):
    pass
