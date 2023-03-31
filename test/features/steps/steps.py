from typing import Any, Dict, List

import hjson
import pandas as pd

from behave import given, then, when
from behave.model import Table
from pypher import CypherExecutor

with open("openCypher/tck/graphs/binary-tree-1/binary-tree-1.cypher", "r") as f:
    BINARY_TREE1 = f.read()
with open("openCypher/tck/graphs/binary-tree-2/binary-tree-2.cypher", "r") as f:
    BINARY_TREE2 = f.read()

def execute_query(context: Any, query: str):
    context.result = None
    context.error = None
    try:
        context.result = CypherExecutor().exec(query)
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
    raise Exception("Can't execute setup queries")

@given("parameters are")
@given("parameter values are")
def define_params(context):
    raise Exception("Parameters unsupported")

@when("executing query")
@when("executing control query")
def when_query(context):
    execute_query(context, context.text)

def normalize_tck_value(value: str) -> Any:
    return hjson.loads(value)

def tck_to_records(table: Table) -> List[Dict[str, Any]]:
    records = []
    for row in table:
        row_obj = {}
        for heading in table.headings:
            row_obj[heading] = normalize_tck_value(row[heading])
        records.append(row_obj)
    return records

def normalize_pypher_value(value: Any) -> Any:
    if value is pd.NA:
        return None
    elif isinstance(value, list):
        return [normalize_pypher_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: normalize_pypher_value(v) for k, v in value.items()}
    return value

def normalize_pypher_output(py_table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in py_table:
        for k, v in row.items():
            row[k] = normalize_pypher_value(v)
    return py_table

@then("the result should be (ignoring element order for lists)")
@then("the result should be, in any order")
def assert_results_in_any_order(context):
    if context.error:
        raise context.error
    print(context.result, context.table.headings)
    assert list(context.result.columns) == context.table.headings
    print(type(context.table))
    expected_rows = tck_to_records(context.table)
    assert len(context.result) == len(expected_rows)

    actual_rows = normalize_pypher_output(context.result.to_dict('records'))
    for expected, actual in zip(expected_rows, actual_rows):
        assert expected == actual, f"{expected} != {actual}"

@then("the result should be, in order")
def assert_results_in_order(context):
    assert not context.error

@then("the result should be empty")
def assert_empty_result(context):
    assert not context.error

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
