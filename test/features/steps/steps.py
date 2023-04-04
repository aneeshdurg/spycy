from typing import Any, Dict, List, Optional, Tuple

import hjson
import pandas as pd
from behave import given, then, when
from behave.model import Table

from antlr4 import *
from spycy import pattern_graph
from spycy.gen.CypherLexer import CypherLexer
from spycy.gen.CypherParser import CypherParser
from spycy.spycy import CypherExecutor

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
    raise Exception("Parameters unsupported")


@when("executing query")
@when("executing control query")
def when_query(context):
    execute_query(context, context.text)


def parse_type(value: str) -> Tuple[Optional[str], str]:
    if value[0] != ":":
        return None, value
    remainder = value[1:]
    type_ = ""
    while remainder[0] not in ["{", ")", "]", ":"]:
        type_ += remainder[0]
        remainder = remainder[1:]
    if len(type_) == 0:
        raise ValueError("Could not parse type")
    return type_, remainder


def parse_props(value: str) -> Tuple[Any, str]:
    if value[0] != "{":
        return None, value
    remainder = value[1:]
    props_str = value[0]
    balance = 1
    while remainder[0] not in [")", "]", " "]:
        props_str += remainder[0]
        remainder = remainder[1:]
        if remainder == "{":
            balance += 1
        if remainder == "}":
            balance -= 1
        if balance == 0:
            break

    if balance != 0:
        raise ValueError("Could not parse props")

    return hjson.parse(props_str), remainder


def parse_tck_node(context, value: str) -> Any:
    input_stream = InputStream(value)
    lexer = CypherLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = CypherParser(stream)
    root = parser.oC_Pattern()

    pgraph = context.executor._interpret_pattern(root)
    assert len(pgraph.nodes) == 1
    return list(pgraph.nodes.values())[0]


def normalize_tck_value(context, value: str) -> Any:
    if value.startswith("("):
        return parse_tck_node(context, value)
    if value.startswith("["):
        try:
            return parse_tck_edge_details(context, value)
        except Exception:
            pass
    if value.startswith("<"):
        try:
            return parse_tck_path(value)
        except ValueError:
            pass
    return hjson.loads(value)


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
    elif isinstance(value, CypherExecutor.Node):
        data_node = context.executor.graph.nodes[value.id_]
        id_ = pattern_graph.NodeID(0)
        return pattern_graph.Node(id_, None, data_node["labels"], None)
    return value


def normalize_spycy_output(
    context, py_table: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    for row in py_table:
        for k, v in row.items():
            row[k] = normalize_spycy_value(context, v)
    return py_table


@then("the result should be (ignoring element order for lists)")
@then("the result should be, in any order")
def assert_results_in_any_order(context):
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
def assert_error(context, errorType, error):
    assert context.error is not None


@then("the side effects should be")
def assert_side_effects(context):
    pass


@then("no side effects")
def assert_no_side_effects(context):
    pass
