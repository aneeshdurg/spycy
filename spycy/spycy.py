#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from antlr4.error.ErrorListener import ErrorListener

from antlr4 import *
from spycy import matcher, pattern_graph
from spycy.errors import ExecutionError
from spycy.functions import function_registry, is_aggregation
from spycy.gen.CypherLexer import CypherLexer
from spycy.gen.CypherParser import CypherParser
from spycy.types import Edge, FunctionContext, Node
from spycy.visitor import hasType, visitor


@dataclass
class GeneratorErrorListener(ErrorListener):
    errors_caught: int = 0

    def syntaxError(self, recognizer, offendingSymbol, line, col, msg, e):
        print(
            "Syntax error at line {} col {}: {}".format(line, col, msg), file=sys.stderr
        )
        self.errors_caught += 1


@dataclass
class CypherExecutor:
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame([{" ": 0}]))
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    # TODO use _table_accesses to speed up CREATE/MATCH
    _table_accesses: int = 0
    _returned: bool = False

    _deleted_ids: Set[int] = field(default_factory=set)

    _parameters: Dict[str, str] = field(default_factory=dict)

    def set_params(self, parameters: Dict[str, str]):
        self._parameters = parameters

    def _has_aggregation(self, ctx) -> bool:
        found_aggregation = False

        def is_aggregation_visitor(ctx) -> bool:
            nonlocal found_aggregation
            if isinstance(ctx, CypherParser.OC_AtomContext):
                assert ctx.children
                count = ctx.children[0].getText().lower() == "count"
                if count:
                    found_aggregation = True
                    return False
            if isinstance(ctx, CypherParser.OC_FunctionInvocationContext):
                fnname = ctx.oC_FunctionName()
                assert fnname
                fnname = fnname.getText()
                if is_aggregation(fnname):
                    found_aggregation = True
                    return False
            return True

        visitor(ctx, is_aggregation_visitor)
        return found_aggregation

    def _vend_node_id(self) -> int:
        if len(self._deleted_ids):
            return self._deleted_ids.pop()
        return len(self.graph.nodes)

    def reset_table(self):
        self.table = pd.DataFrame([{" ": 0}])

    def table_to_json(self):
        def make_serializable(x):
            if x is pd.NA:
                return None

            if isinstance(x, Node):
                return {"type": "Node", "id": x.id_}
            elif isinstance(x, Edge):
                return {"type": "Edge", "id": x.id_}
            elif isinstance(x, dict):
                for k, v in x.items():
                    x[k] = make_serializable(v)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    x[i] = make_serializable(v)
            elif np.issubdtype(type(x), np.integer):
                return int(x)
            elif np.issubdtype(type(x), np.bool_):
                return bool(x)
            elif np.issubdtype(type(x), np.float_):
                return float(x)
            return x

        rows = self.table.to_dict("records")
        rows = [make_serializable(row) for row in rows]
        return json.dumps(rows)

    def _evaluate_list_literal(
        self, expr: CypherParser.OC_ListLiteralContext
    ) -> pd.Series:
        rows = len(self.table)
        data = []
        for _ in range(rows):
            data.append([])
        elems = expr.oC_Expression()
        assert elems is not None
        for list_el in elems:
            values = self._evaluate_expression(list_el)
            for i, l in enumerate(data):
                l.append(values[i])
        return pd.Series(data)

    def _evaluate_map_literal(
        self, expr: CypherParser.OC_MapLiteralContext
    ) -> pd.Series:
        rows = len(self.table)
        data = []
        for _ in range(rows):
            data.append({})
        prop_key_names = expr.oC_PropertyKeyName()
        if prop_key_names:
            values = expr.oC_Expression()
            for key_name, value_expr in zip(prop_key_names, values):
                key = key_name.getText()
                value = self._evaluate_expression(value_expr)
                for i, m in enumerate(data):
                    m[key] = value[i]
        return pd.Series(data)

    def _evaluate_literal(self, expr: CypherParser.OC_LiteralContext) -> pd.Series:
        if list_literal := expr.oC_ListLiteral():
            return self._evaluate_list_literal(list_literal)

        if map_literal := expr.oC_MapLiteral():
            return self._evaluate_map_literal(map_literal)

        rows = len(self.table)
        dtype: Any = None
        value: Any = None
        if expr.getText().lower() == "null":
            value = pd.NA
        elif expr.getText().lower() == "true":
            value = True
            dtype = bool
        elif expr.getText().lower() == "false":
            value = False
            dtype = bool
        elif number := expr.oC_NumberLiteral():
            dtype = "float64"
            nstr = number.getText()
            if (int_lit := number.oC_IntegerLiteral()) and "e" not in nstr:
                dtype = "int64"
            value = eval(nstr)
            if math.isinf(value):
                raise ExecutionError("SyntaxError::FloatingPointOverflow")
        elif expr.StringLiteral():
            value = eval(expr.getText())
            dtype = str

        assert value is not None, "Unsupported literal type"
        data = [value] * rows
        if dtype:
            return pd.Series(data, dtype=dtype)
        return pd.Series(data)

    def _evaluate_function_invocation(
        self, expr: CypherParser.OC_FunctionInvocationContext
    ) -> pd.Series:
        fnname = expr.oC_FunctionName()
        assert fnname
        fnname = fnname.getText()

        assert expr.children
        is_distinct = False
        if expr.children[2].getText().lower() == "distinct":
            is_distinct = True

        params = []
        if param_exprs := expr.oC_Expression():
            for param_expr in param_exprs:
                params.append(self._evaluate_expression(param_expr))
        fnctx = FunctionContext(self.table, self.graph)
        return function_registry(fnname, params, fnctx)

    def _do_evaluate_list_comp(
        self,
        filter_expr: CypherParser.OC_FilterExpressionContext,
        list_expr: Optional[CypherParser.OC_ExpressionContext],
        fill_value=None,
        aggregate=lambda x: x.to_list(),
    ) -> Tuple[pd.Series, pd.Series]:
        if len(self.table) == 0:
            return (pd.Series([], dtype=object), pd.Series([], dtype=object))

        if fill_value is None:
            fill_value = []

        id_in_coll = filter_expr.oC_IdInColl()
        assert id_in_coll
        id_to_bind = id_in_coll.oC_Variable().getText()
        coll = id_in_coll.oC_Expression()

        column = self._evaluate_expression(coll)

        where_expr = filter_expr.oC_Where()
        if where_expr:
            where_expr = where_expr.oC_Expression()

        old_table = self.table.copy()
        self.table[id_to_bind] = column
        self.table["!__internal_index"] = list(range(len(self.table)))

        def check_for_empty_list(x):
            if not isinstance(x, list):
                raise ExecutionError(f"TypeError::Expected list, got {type(x)}")
            return len(x) > 0

        mask = [
            check_for_empty_list(self.table[id_to_bind][i])
            for i in range(len(self.table))
        ]
        self.table = self.table[mask]
        self.table = self.table.explode(id_to_bind, ignore_index=True)
        if where_expr:
            self._filter_table(where_expr)
        if list_expr:
            new_col = self._evaluate_expression(list_expr)
        else:
            new_col = self.table[id_to_bind]
        tmp_table = pd.DataFrame()
        tmp_table["result"] = new_col
        tmp_table["idx"] = self.table["!__internal_index"]
        tmp_table = tmp_table.groupby(["idx"]).agg({"result": aggregate})
        # We might have holes from keys that got completely filtered out
        if len(tmp_table) == 0:
            tmp_table["result"] = pd.Series([fill_value] * len(old_table))
        else:
            tmp_table = tmp_table.reindex(range(len(old_table)), fill_value=fill_value)
        output_column = tmp_table["result"]
        self.table = old_table

        return column, output_column

    def _evaluate_list_comp(
        self, expr: CypherParser.OC_ListComprehensionContext
    ) -> pd.Series:
        filter_expr = expr.oC_FilterExpression()
        assert filter_expr
        list_expr = expr.oC_Expression()
        return self._do_evaluate_list_comp(filter_expr, list_expr)[1]

    def _evaluate_parameter(self, expr: CypherParser.OC_ParameterContext) -> pd.Series:
        name_expr = expr.oC_SymbolicName()
        assert name_expr
        name = name_expr.getText()

        if name not in self._parameters:
            raise ExecutionError("ParameterNotFound")

        old_table = self.table
        self.reset_table()
        ast = self._getAST(
            self._parameters[name], get_root=lambda parser: parser.oC_Expression()
        )
        assert ast
        col = self._evaluate_expression(ast)
        assert len(col) == 1
        value = col[0]
        self.table = old_table
        return pd.Series([value] * len(self.table))

    def _evaluate_case_alternatives(
        self, alt_exprs: List[CypherParser.OC_CaseAlternativeContext]
    ) -> List[Tuple[pd.Series, pd.Series]]:
        alternatives = []
        for alt_expr in alt_exprs:
            alt_sub_exprs = alt_expr.oC_Expression()
            assert alt_sub_exprs
            assert len(alt_sub_exprs) == 2
            [when_expr, then_expr] = alt_sub_exprs
            when_col = self._evaluate_expression(when_expr)
            then_col = self._evaluate_expression(then_expr)
            alternatives.append((when_col, then_col))
        return alternatives

    def _evaluate_case_generic(
        self, expr: CypherParser.OC_CaseExpressionContext
    ) -> pd.Series:
        # This is what the openCypher docs refer to as the 'generic' form of CASE
        else_ = None
        exprs = expr.oC_Expression()
        if exprs:
            else_ = self._evaluate_expression(exprs[0])

        alt_exprs = expr.oC_CaseAlternative()
        assert alt_exprs
        alternatives = self._evaluate_case_alternatives(alt_exprs)

        output = []
        for i in range(len(self.table)):
            found = False
            for (when_col, then_col) in alternatives:
                when_val = when_col[i]
                if when_val is pd.NA:
                    continue
                if when_val == True:
                    output.append(then_col[i])
                    found = True
                    break
            if found:
                continue
            if else_ is not None:
                output.append(else_[i])
            else:
                raise ExecutionError("Non exhaustive case pattern")
        return pd.Series(output)

    def _evaluate_case(self, expr: CypherParser.OC_CaseExpressionContext) -> pd.Series:
        assert expr.children
        test_expr = None
        for child in expr.children:
            if isinstance(child, CypherParser.OC_ExpressionContext):
                test_expr = child
            elif child.getText().lower() == "else":
                break
        if not test_expr:
            return self._evaluate_case_generic(expr)
        assert test_expr

        test = self._evaluate_expression(test_expr)

        else_ = None
        exprs = expr.oC_Expression()
        assert exprs
        if len(exprs) == 2:
            else_ = self._evaluate_expression(exprs[1])

        alt_exprs = expr.oC_CaseAlternative()
        assert alt_exprs
        alternatives = self._evaluate_case_alternatives(alt_exprs)

        output = []
        for i, test_val in enumerate(test):
            found = False
            if test_val is pd.NA:
                for (when_col, then_col) in alternatives:
                    if when_col[i] is pd.NA:
                        output.append(then_col[i])
                        found = True
                        break
                if found:
                    continue
            else:
                for (when_col, then_col) in alternatives:
                    when_val = when_col[i]
                    if when_val is pd.NA:
                        continue
                    if when_val == test_val:
                        output.append(then_col[i])
                        found = True
                        break
                if found:
                    continue
            if else_ is not None:
                output.append(else_[i])
            else:
                raise ExecutionError("Non exhaustive case pattern")
        return pd.Series(output)

    def _evaluate_quantifier(
        self, expr: CypherParser.OC_QuantifierContext
    ) -> pd.Series:
        filter_expr = expr.oC_FilterExpression()
        assert filter_expr

        is_all = False
        # vacuously true
        fill_value = True

        assert expr.children
        if expr.children[0].getText().lower() == "all":
            aggregate = lambda x: len(x)
            is_all = True
            fill_value = 0
        elif expr.children[0].getText().lower() == "any":
            aggregate = lambda x: len(x) > 0
            # vacuously false
            fill_value = False
        elif expr.children[0].getText().lower() == "none":
            aggregate = lambda x: len(x) == 0
        elif expr.children[0].getText().lower() == "single":
            aggregate = lambda x: len(x) == 1
            # vacuously false
            fill_value = False
        else:
            raise ExecutionError("Unsupported quantifier")

        column, output = self._do_evaluate_list_comp(
            filter_expr, None, fill_value, aggregate
        )
        if is_all:
            return pd.Series(len(in_) == out for in_, out in zip(column, output))
        return output

    def _evaluate_atom(self, expr: CypherParser.OC_AtomContext) -> pd.Series:
        if literal := expr.oC_Literal():
            return self._evaluate_literal(literal)
        if parameter := expr.oC_Parameter():
            return self._evaluate_parameter(parameter)
        if case_ := expr.oC_CaseExpression():
            return self._evaluate_case(case_)
        if list_comp := expr.oC_ListComprehension():
            return self._evaluate_list_comp(list_comp)
        if pattern_comp := expr.oC_PatternComprehension():
            return self._evaluate_pattern_comp(pattern_comp)
        if rels := expr.oC_Quantifier():
            return self._evaluate_quantifier(rels)
        if par_expr := expr.oC_ParenthesizedExpression():
            return self._evaluate_expression(par_expr.oC_Expression())
        if func_call := expr.oC_FunctionInvocation():
            return self._evaluate_function_invocation(func_call)
        if existential_subquery := expr.oC_ExistentialSubquery():
            return self._evaluate_existential_subquery(existential_subquery)
        if variable := expr.oC_Variable():
            self._table_accesses += 1
            # assert not variable.EscapedSymbolicName(), "Unsupported query - variable in `` unsupported"
            name = variable.getText()
            if name not in self.table:
                raise ExecutionError(f"SyntaxError::UnknownVariable {name}")
            return self.table[name]

        assert expr.children
        operation = expr.children[0].getText()
        if operation == "count":
            return pd.Series([len(self.table)])
        raise AssertionError(f"Operation {operation} unsupported")

    def _evaluate_list_op(
        self, lhs: pd.Series, expr: CypherParser.OC_ListOperatorExpressionContext
    ) -> pd.Series:
        pre_range_accessor = None
        post_range_accessor = None
        start = False
        found_accessor = False
        assert expr.children
        for child in expr.children:
            if not start:
                if child.getText() == "[":
                    start = True
                    continue
            elif child.getText() == "]":
                break
            else:
                if isinstance(child, CypherParser.OC_ExpressionContext):
                    if found_accessor:
                        post_range_accessor = child
                    else:
                        pre_range_accessor = child
                else:
                    found_accessor = True
        if found_accessor:
            if not pre_range_accessor and not post_range_accessor:
                return lhs
            if pre_range_accessor and post_range_accessor:
                pre_eval = self._evaluate_expression(pre_range_accessor)
                post_eval = self._evaluate_expression(post_range_accessor)
                return pd.Series(
                    e[pre_eval[i] : post_eval[i]] for i, e in enumerate(lhs)
                )
        else:
            assert pre_range_accessor
            rhs = self._evaluate_expression(pre_range_accessor)
            output = []
            for l, r in zip(lhs, rhs):
                if l is pd.NA:
                    output.append(pd.NA)
                elif isinstance(l, Node):
                    if not isinstance(r, str):
                        raise ExecutionError("TypeError::Map access with non-str index")
                    props = self.graph.nodes[l.id_]["properties"]
                    output.append(props.get(r, pd.NA))
                elif isinstance(l, Edge):
                    if not isinstance(r, str):
                        raise ExecutionError("TypeError::Map access with non-str index")
                    props = self.graph.edges[l.id_]["properties"]
                    output.append(props.get(r, pd.NA))
                elif isinstance(l, dict):
                    if not isinstance(r, str):
                        raise ExecutionError("TypeError::Map access with non-str index")
                    output.append(l.get(r, pd.NA))
                elif isinstance(l, list):
                    if not np.issubdtype(type(r), np.integer):
                        raise ExecutionError(
                            "TypeError::List access with non-integer index"
                        )
                    output.append(l[r])
                else:
                    raise ExecutionError(
                        f"TypeError::subscript access on non-subscriptable type {type(l)}"
                    )
            return pd.Series(output)

    def _evaluate_property_lookup(
        self, lhs: pd.Series, expr: CypherParser.OC_PropertyLookupContext
    ) -> pd.Series:
        if len(lhs) == 0:
            return pd.Series([])

        el = list(lhs)[0]
        output = []
        key_expr = expr.oC_PropertyKeyName()
        assert key_expr
        key = key_expr.getText()

        for el in lhs:
            if el is pd.NA:
                output.append(pd.NA)
            elif isinstance(el, Node):
                output.append(self.graph.nodes[el.id_]["properties"].get(key, pd.NA))
            elif isinstance(el, Edge):
                output.append(self.graph.edges[el.id_]["properties"].get(key, pd.NA))
            elif isinstance(el, dict):
                output.append(el.get(key, pd.NA))
            else:
                raise ExecutionError("TypeError::InvalidPropertyAccess")
        return pd.Series(output)

    def _evaluate_node_labels(
        self, lhs: pd.Series, labels: CypherParser.OC_NodeLabelsContext
    ) -> pd.Series:
        label_exprs = labels.oC_NodeLabel()
        assert label_exprs
        label_vals = set()
        for label in label_exprs:
            name = label.oC_LabelName()
            assert name
            label_vals.add(name.getText())

        output = []
        for el in lhs:
            if el is pd.NA:
                output.append(pd.NA)
            elif isinstance(el, Node):
                node_data = self.graph.nodes[el.id_]
                output.append(all(l in node_data["labels"] for l in label_vals))
            elif isinstance(el, Edge):
                node_data = self.graph.edges[el.id_]
                output.append(all(l == node_data["type"] for l in label_vals))
            else:
                raise ExecutionError("TypeError - labels requires Node or Edge")
        return pd.Series(output)

    def _evaluate_non_arithmetic_operator(
        self, expr: CypherParser.OC_NonArithmeticOperatorExpressionContext
    ) -> pd.Series:
        atom = expr.oC_Atom()
        assert atom
        lhs = self._evaluate_atom(atom)

        assert expr.children
        for child in expr.children[1:]:
            if isinstance(child, CypherParser.OC_ListOperatorExpressionContext):
                lhs = self._evaluate_list_op(lhs, child)

            if isinstance(child, CypherParser.OC_PropertyLookupContext):
                lhs = self._evaluate_property_lookup(lhs, child)

            if isinstance(child, CypherParser.OC_NodeLabelsContext):
                return self._evaluate_node_labels(lhs, child)
        return lhs

    def _evaluate_unary_add_or_sub(
        self, expr: CypherParser.OC_UnaryAddOrSubtractExpressionContext
    ) -> pd.Series:
        assert expr.children
        negate = expr.children[0].getText() == "-"

        child = expr.oC_NonArithmeticOperatorExpression()
        assert child
        output = self._evaluate_non_arithmetic_operator(child)
        if negate:
            return -1 * output
        return output

    def _evaluate_bin_op(self, lhs: pd.Series, rhs: pd.Series, op: str) -> pd.Series:
        def evaluate_equality(v1, v2):
            if v1 is pd.NA or v2 is pd.NA:
                return pd.NA
            if isinstance(v1, list):
                if not isinstance(v2, list):
                    return False
                if len(v1) > len(v2):
                    return False
                for e1, e2 in zip(v1, v2):
                    eq = evaluate_equality(e1, e2)
                    if eq is pd.NA or not eq:
                        return eq
                if len(v1) < len(v2):
                    return False
                return True
            if isinstance(v1, dict):
                if not isinstance(v2, dict):
                    return False
                if set(v1.keys()) != set(v2.keys()):
                    return False
                for k, e1 in v1.items():
                    e2 = v2[k]
                    eq = evaluate_equality(e1, e2)
                    if eq is pd.NA or not eq:
                        return eq
                return True
            return v1 == v2

        if op == "=":
            return pd.Series(evaluate_equality(l, r) for l, r in zip(lhs, rhs))
        if op == "<>":

            def invert(x):
                if x is pd.NA:
                    return x
                return not x

            return pd.Series(invert(evaluate_equality(l, r)) for l, r in zip(lhs, rhs))

        if op == "^":
            op = "**"
        op_f = eval(f"lambda x, y: x {op} y")

        output = []
        for l, r in zip(lhs, rhs):
            try:
                output.append(op_f(l, r))
            except ZeroDivisionError:
                output.append(math.nan)
            except Exception as e:
                output.append(pd.NA)
        output = pd.Series(output)
        return output

    def _evaluate_power_of(
        self, expr: CypherParser.OC_PowerOfExpressionContext
    ) -> pd.Series:
        assert expr.children
        lhs = self._evaluate_unary_add_or_sub(expr.children[0])

        ops = ["^"]
        last_op = None
        for child in expr.children[1:]:
            if (op := child.getText()) in ops:
                last_op = op
            elif isinstance(child, CypherParser.OC_UnaryAddOrSubtractExpressionContext):
                assert last_op
                rhs = self._evaluate_unary_add_or_sub(child)
                lhs = self._evaluate_bin_op(lhs, rhs, last_op)
        return lhs

    def _evaluate_multiply_divide_modulo(
        self, expr: CypherParser.OC_MultiplyDivideModuloExpressionContext
    ) -> pd.Series:
        assert expr.children
        lhs = self._evaluate_power_of(expr.children[0])

        ops = ["*", "/", "%"]
        last_op = None
        for child in expr.children[1:]:
            if (op := child.getText()) in ops:
                last_op = op
            elif isinstance(child, CypherParser.OC_PowerOfExpressionContext):
                assert last_op
                rhs = self._evaluate_power_of(child)
                lhs = self._evaluate_bin_op(lhs, rhs, last_op)
        return lhs

    def _evaluate_add_or_subtract(
        self, expr: CypherParser.OC_AddOrSubtractExpressionContext
    ) -> pd.Series:
        assert expr.children
        lhs = self._evaluate_multiply_divide_modulo(expr.children[0])

        ops = ["+", "-"]
        last_op = None
        for child in expr.children[1:]:
            if (op := child.getText()) in ops:
                last_op = op
            elif isinstance(
                child, CypherParser.OC_MultiplyDivideModuloExpressionContext
            ):
                assert last_op
                rhs = self._evaluate_multiply_divide_modulo(child)
                if last_op == "+":
                    output = []
                    for l, r in zip(lhs, rhs):
                        if l is pd.NA or r is pd.NA:
                            output.append(pd.NA)
                        elif isinstance(l, list):
                            if isinstance(r, list):
                                output.append(l + r)
                            else:
                                output.append(l + [r])
                        elif isinstance(r, list):
                            output.append([l] + r)
                        else:
                            output.append(l + r)
                    lhs = pd.Series(output)
                else:
                    lhs = self._evaluate_bin_op(lhs, rhs, last_op)

        return lhs

    def _evaluate_string_op(
        self, lhs: pd.Series, expr: CypherParser.OC_StringPredicateExpressionContext
    ) -> pd.Series:
        is_startswith = False
        is_endswith = False
        is_contains = False
        assert expr.children
        for child in expr.children:
            if child.getText().lower() == "starts":
                is_startswith = True
                break
            if child.getText().lower() == "ends":
                is_endswith = True
                break
            if child.getText().lower() == "contains":
                is_contains = True
                break
        add_or_sub_expr = expr.oC_AddOrSubtractExpression()
        assert add_or_sub_expr
        rhs = self._evaluate_add_or_subtract(add_or_sub_expr)

        def startswith(i: int, x: str):
            rhs_e: str = rhs[i]
            return x.startswith(rhs_e)

        def endswith(i: int, x: str):
            rhs_e: str = rhs[i]
            return x.endswith(rhs_e)

        def contains(i: int, x: str):
            rhs_e: str = rhs[i]
            return rhs_e in x

        op = None
        if is_startswith:
            op = startswith
        elif is_endswith:
            op = endswith
        elif is_contains:
            op = contains
        assert op
        return pd.Series(op(i, e) for i, e in enumerate(lhs))

    def _evaluate_list_predicate(
        self, lhs: pd.Series, expr: CypherParser.OC_ListPredicateExpressionContext
    ) -> pd.Series:
        add_or_sub_expr = expr.oC_AddOrSubtractExpression()
        assert add_or_sub_expr
        rhs = self._evaluate_add_or_subtract(add_or_sub_expr)
        # This is an IN expression
        def contains(list_, element):
            if not isinstance(list_, list):
                raise ExecutionError("SyntaxError::InvalidArgumentType")
            for item in list_:
                if isinstance(element, list):
                    if isinstance(item, list):
                        return element == item
                else:
                    if element == item:
                        return True
            return False

        return pd.Series(contains(e, lhs[i]) for i, e in enumerate(rhs))

    def _evaluate_null_predicate(
        self, lhs: pd.Series, expr: CypherParser.OC_NullPredicateExpressionContext
    ) -> pd.Series:

        result = lhs.apply(lambda x: x is pd.NA)
        if any(x.getText().lower() == "not" for x in expr.children):
            result = ~result
        return result

    def _evaluate_string_list_null_predicate(
        self, expr: CypherParser.OC_StringListNullPredicateExpressionContext
    ) -> pd.Series:
        add_or_sub_expr = expr.oC_AddOrSubtractExpression()
        assert add_or_sub_expr
        output = self._evaluate_add_or_subtract(add_or_sub_expr)

        assert expr.children
        for child in expr.children[1:]:
            if isinstance(child, CypherParser.OC_StringPredicateExpressionContext):
                output = self._evaluate_string_op(output, child)
            if isinstance(child, CypherParser.OC_ListPredicateExpressionContext):
                output = self._evaluate_list_predicate(output, child)
            if isinstance(child, CypherParser.OC_NullPredicateExpressionContext):
                output = self._evaluate_null_predicate(output, child)
        return output

    def _evaluate_comparision(
        self, expr: CypherParser.OC_ComparisonExpressionContext
    ) -> pd.Series:
        add_or_sub = expr.oC_StringListNullPredicateExpression()
        assert add_or_sub
        lhs = self._evaluate_string_list_null_predicate(add_or_sub)

        p_comps = expr.oC_PartialComparisonExpression()
        if not p_comps:
            return lhs

        for p_comp in p_comps:
            rhs_expr = p_comp.oC_StringListNullPredicateExpression()
            assert rhs_expr
            rhs = self._evaluate_string_list_null_predicate(rhs_expr)
            lhs = self._evaluate_bin_op(lhs, rhs, p_comp.children[0].getText())
        return lhs

    def _evaluate_not(self, expr: CypherParser.OC_NotExpressionContext) -> pd.Series:
        assert expr.children
        not_count = sum(1 for e in expr.children if e.getText().lower() == "not")
        negate = not_count % 2 == 1

        comp_expr = expr.oC_ComparisonExpression()
        assert comp_expr
        output = self._evaluate_comparision(comp_expr)
        if negate:

            def invert(x):
                if x is pd.NA:
                    return pd.NA
                if not np.issubdtype(type(x), np.bool_):
                    raise ExecutionError(
                        f"TypeError::Not expected boolean got {type(x)}"
                    )
                return not x

            return output.apply(lambda x: invert(x))
        return output

    def _evaluate_and(self, expr: CypherParser.OC_AndExpressionContext) -> pd.Series:
        expressions = expr.oC_NotExpression()
        assert expressions

        lhs = self._evaluate_not(expressions[0])
        for not_expr in expressions[1:]:
            output = []
            rhs = self._evaluate_not(not_expr)
            for l, r in zip(lhs, rhs):
                if l is not pd.NA and not np.issubdtype(type(l), np.bool_):
                    raise ExecutionError(
                        f"TypeError - and expected boolean got {type(l)}"
                    )
                if r is not pd.NA and not np.issubdtype(type(r), np.bool_):
                    raise ExecutionError(
                        f"TypeError - and expected boolean got {type(r)}"
                    )

                if l is pd.NA and r is pd.NA:
                    output.append(pd.NA)
                elif l is pd.NA or r is pd.NA:
                    if l is True or r is True:
                        output.append(pd.NA)
                    else:
                        output.append(False)
                else:
                    output.append(l & r)
            lhs = pd.Series(output, dtype=object)
        return lhs

    def _evaluate_xor(self, expr: CypherParser.OC_XorExpressionContext) -> pd.Series:
        expressions = expr.oC_AndExpression()
        assert expressions

        lhs = self._evaluate_and(expressions[0])
        for and_expr in expressions[1:]:
            output = []
            rhs = self._evaluate_and(and_expr)
            for l, r in zip(lhs, rhs):
                if l is not pd.NA and not np.issubdtype(type(l), np.bool_):
                    raise ExecutionError(
                        f"TypeError - or expected boolean got {type(l)}"
                    )
                if r is not pd.NA and not np.issubdtype(type(r), np.bool_):
                    raise ExecutionError(
                        f"TypeError - or expected boolean got {type(r)}"
                    )

                if l is pd.NA or r is pd.NA:
                    output.append(pd.NA)
                else:
                    output.append(l ^ r)
            lhs = pd.Series(output, dtype=object)
        return lhs

    def _evaluate_or(self, expr: CypherParser.OC_OrExpressionContext) -> pd.Series:
        expressions = expr.oC_XorExpression()
        assert expressions

        lhs = self._evaluate_xor(expressions[0])
        for xor_expr in expressions[1:]:
            output = []
            rhs = self._evaluate_xor(xor_expr)
            for l, r in zip(lhs, rhs):
                if l is not pd.NA and not np.issubdtype(type(l), np.bool_):
                    raise ExecutionError(
                        f"TypeError - or expected boolean got {type(l)}"
                    )
                if r is not pd.NA and not np.issubdtype(type(r), np.bool_):
                    raise ExecutionError(
                        f"TypeError - or expected boolean got {type(r)}"
                    )

                if l is pd.NA and r is pd.NA:
                    output.append(pd.NA)
                elif l is pd.NA or r is pd.NA:
                    if l is False or r is False:
                        output.append(pd.NA)
                    else:
                        output.append(True)
                else:
                    output.append(l | r)
            lhs = pd.Series(output, dtype=object)
        return lhs

    def _evaluate_expression(
        self, expr: CypherParser.OC_ExpressionContext
    ) -> pd.Series:
        """Returns a DataFrame with a single column"""
        or_expr = expr.oC_OrExpression()
        assert or_expr
        return self._evaluate_or(or_expr)

    def _process_order(self, node: CypherParser.OC_OrderContext):
        sort_keys = []
        sort_asc = []
        keys_to_remove = set()

        sort_items = node.oC_SortItem()
        assert sort_items
        for sort_item in sort_items:
            expr = sort_item.oC_Expression()
            assert expr
            alias = expr.getText()
            if alias not in self.table:
                if self._has_aggregation(expr):
                    raise ExecutionError(
                        "SyntaxError::InvalidAggregation - cannot aggregate during ORDER BY"
                    )
                col = self._evaluate_expression(expr)
                self.table[alias] = col
                keys_to_remove.add(alias)
            sort_keys.append(alias)

            if len(sort_item.children) > 2:
                asc = sort_item.children[2].getText().lower().startswith("asc")
                sort_asc.append(asc)
            else:
                sort_asc.append(True)

        # TODO this actually needs a much more nuanced approach where
        # we build up the table by using row-wise comparisions.
        def key(col):
            def no_NA(x):
                if x is pd.NA:
                    return math.inf
                if isinstance(x, list):
                    return (len(x), [no_NA(e) for e in x])
                if isinstance(x, dict):
                    return {k: no_NA(v) for k, v in x.items()}
                return x

            return pd.Series([no_NA(x) for x in col])

        self.table.sort_values(
            sort_keys,
            ascending=sort_asc,
            ignore_index=True,
            inplace=True,
            key=key,
        )
        for key in keys_to_remove:
            del self.table[key]

    def _process_skip(self, node: CypherParser.OC_SkipContext):
        old_table = self.table

        self.reset_table()
        expr = node.oC_Expression()
        assert expr
        skip = self._evaluate_expression(expr)
        assert len(skip) == 1
        skip = skip[0]
        assert np.issubdtype(type(skip), np.integer)

        if skip == 0:
            self.table = old_table
            return
        if skip < 0:
            raise ExecutionError("SyntaxError::NegativeIntegerArgument in skip")

        self.table = old_table.tail(-skip)
        self.table.reset_index(drop=True, inplace=True)

    def _process_limit(self, node: CypherParser.OC_LimitContext):
        old_table = self.table

        self.reset_table()
        expr = node.oC_Expression()
        assert expr
        limit = self._evaluate_expression(expr)
        assert len(limit) == 1
        limit = limit[0]

        assert np.issubdtype(type(limit), np.integer)
        if limit < 0:
            raise ExecutionError("SyntaxError::NegativeIntegerArgument in limit")

        self.table = old_table.head(limit)

    def _process_projection_body(
        self, node: CypherParser.OC_ProjectionBodyContext, is_with: bool = False
    ):
        is_distinct = node.DISTINCT()
        assert not is_distinct, "Unsupported query - DISTINCT not implemented"

        proj_items = node.oC_ProjectionItems()
        assert proj_items

        if proj_items.children[0].getText() == "*":
            assert len(proj_items.children) == 1
            return

        def get_alias(ctx: CypherParser.OC_ProjectionItemContext):
            var = ctx.oC_Variable()
            if var:
                alias = var.getText()
            else:
                expr = proj.oC_Expression()
                if is_with:
                    var_expr = []

                    def get_var_expr(f):
                        if isinstance(f, CypherParser.OC_VariableContext):
                            var_expr.append(f)
                        return True

                    visitor(expr, get_var_expr)
                    if len(var_expr) != 1 or var_expr[0].getText() != expr.getText():
                        raise ExecutionError("SyntaxError::NoExpressionAlias")
                alias = expr.getText()
            return alias

        group_by_keys = OrderedDict()
        aggregations = {}
        all_aliases = set()
        for proj in proj_items.oC_ProjectionItem():
            alias = get_alias(proj)
            if self._has_aggregation(proj):
                aggregations[alias] = proj
            else:
                group_by_keys[alias] = proj
            if alias in all_aliases:
                raise ExecutionError("SyntaxError::ColumnNameConflict")
            all_aliases.add(alias)

        output_table = pd.DataFrame()

        if len(aggregations) == 0 or len(group_by_keys) == 0:
            for proj in proj_items.oC_ProjectionItem():
                expr = proj.oC_Expression()
                expr_column = self._evaluate_expression(expr)
                alias = get_alias(proj)
                output_table[alias] = expr_column
        else:
            group_by_columns = OrderedDict()
            for alias, proj in group_by_keys.items():
                expr = proj.oC_Expression()
                group_by_columns[alias] = self._evaluate_expression(expr)

            def get_key(row):
                def get_tuple(value):
                    if isinstance(value, Node):
                        return value.id_
                    if isinstance(value, Edge):
                        return value.id_
                    if isinstance(value, list):
                        return tuple(get_tuple(v) for v in value)
                    if isinstance(value, dict):
                        return tuple((k, get_tuple(v)) for k, v in value.items())
                    return value

                values = []
                for alias in group_by_columns:
                    values.append(group_by_columns[alias][row])
                return get_tuple(values)

            output_keys = OrderedDict()
            for alias in group_by_columns:
                output_keys[alias] = []
            output_keys_row_count = 0

            keys_to_row = {}
            keys_to_subtable = {}
            for i in range(len(self.table)):
                key = get_key(i)
                if key not in keys_to_row:
                    for alias in group_by_columns:
                        output_keys[alias].append(group_by_columns[alias][i])
                    keys_to_row[key] = output_keys_row_count
                    output_keys_row_count += 1
                    keys_to_subtable[key] = pd.DataFrame(
                        [{k: self.table[k][i] for k in self.table}]
                    )
                else:
                    keys_to_subtable[key].loc[
                        len(keys_to_subtable[key])
                    ] = self.table.iloc[i]

            old_table = self.table
            aggregated_columns = {}
            for alias in aggregations:
                aggregated_columns[alias] = [pd.NA] * len(keys_to_row)
            for key, table in keys_to_subtable.items():
                self.table = table
                for alias, proj in aggregations.items():
                    expr = proj.oC_Expression()
                    col = self._evaluate_expression(expr)
                    assert len(col) == 1
                    aggregated_columns[alias][keys_to_row[key]] = col[0]
            self.table = old_table

            output_table = pd.DataFrame()
            for proj in proj_items.oC_ProjectionItem():
                alias = get_alias(proj)
                if alias in output_keys:
                    output_table[alias] = output_keys[alias]
                else:
                    output_table[alias] = aggregated_columns[alias]

        columns_to_remove = []
        if len(aggregations) == 0:
            for col in self.table:
                if col not in output_table:
                    columns_to_remove.append(col)
                    output_table[col] = self.table[col]
        self.table = output_table

        if order := node.oC_Order():
            self._process_order(order)
        if skip := node.oC_Skip():
            self._process_skip(skip)
        if limit := node.oC_Limit():
            self._process_limit(limit)

        for col in columns_to_remove:
            del self.table[col]

    def _process_return(self, node: CypherParser.OC_ReturnContext):
        body = node.oC_ProjectionBody()
        assert body
        self._process_projection_body(body)
        self._returned = True

    def _filter_table(self, filter_expr: CypherParser.OC_ExpressionContext):
        old_columns = self.table.columns
        filter_col = self._evaluate_expression(filter_expr)
        mask = filter_col.fillna(False)
        if len(mask) and mask.dtype.kind != "b":
            raise ExecutionError("TypeError::Expected boolean expression for WHERE")
        new_table = self.table[mask]
        assert new_table is not None
        self.table = new_table
        self.table.reset_index(drop=True, inplace=True)

        if len(self.table) == 0:
            for c in old_columns:
                self.table[c] = []

    def _process_where(self, node: CypherParser.OC_WhereContext):
        old_columns = self.table.columns
        filter_expr = node.oC_Expression()
        assert filter_expr
        self._filter_table(filter_expr)

    def _process_with(self, node: CypherParser.OC_WithContext):
        body = node.oC_ProjectionBody()
        assert body
        self._process_projection_body(body, is_with=True)

        where = node.oC_Where()
        if where:
            self._process_where(where)

    def _process_unwind(self, node: CypherParser.OC_UnwindContext):
        list_expr = node.oC_Expression()
        assert list_expr
        list_column = self._evaluate_expression(list_expr)
        alias_var = node.oC_Variable()
        assert alias_var
        alias = alias_var.getText()
        self.table[alias] = list_column
        self.table = self.table.explode(alias, ignore_index=True)

    def _process_match(self, node: CypherParser.OC_MatchContext):
        assert node.children
        is_optional = node.children[0].getText().lower() == "optional"

        pattern = node.oC_Pattern()
        assert pattern
        pgraph = self._interpret_pattern(pattern)
        node_ids_to_props, edge_ids_to_props = self._evaluate_pattern_graph_properties(
            pgraph
        )

        names_to_data = {}
        for n in pgraph.nodes.values():
            if n.name:
                names_to_data[n.name] = []
        for e in pgraph.edges.values():
            if e.name:
                names_to_data[e.name] = []
        for i in range(len(self.table)):
            m = matcher.Matcher(
                self.graph, pgraph, i, node_ids_to_props, edge_ids_to_props
            )
            initial_state = matcher.MatchResult()
            skip_row = False
            for name in names_to_data:
                if name not in self.table:
                    continue
                found = False
                for _, node_ in pgraph.nodes.items():
                    if node_.name != name:
                        continue
                    found = True
                    value = self.table[name][i]
                    if value is pd.NA:
                        skip_row = True
                    else:
                        if not isinstance(value, Node):
                            raise ExecutionError("TypeError cannot rebind as node")
                        value = value.id_
                    initial_state.node_ids_to_data_ids[node_.id_] = value
                if found:
                    continue

                for _, edge in pgraph.edges.items():
                    if edge.name != name:
                        continue
                    found = True
                    value = self.table[name][i]
                    if value is pd.NA:
                        skip_row = True
                    else:
                        if not isinstance(value, Edge):
                            raise ExecutionError("TypeError cannot rebind as edge")
                        value = value.id_
                    initial_state.edge_ids_to_data_ids[edge.id_] = value
                assert found
            if skip_row:
                results = matcher.MatchResultSet()
            else:
                results = m.match_dfs(initial_state)
            for nid, pnode in pgraph.nodes.items():
                if node_name := pnode.name:
                    data = results.node_ids_to_data_ids.get(nid, [])
                    if is_optional and not data:
                        if node_name in self.table:
                            data = [self.table[node_name][i]]
                        else:
                            data = [pd.NA]
                    else:
                        data = [Node(d) for d in data]
                    names_to_data[node_name].append(data)

            for eid, pedge in pgraph.edges.items():
                if edge_name := pedge.name:
                    data = results.edge_ids_to_data_ids.get(eid, [])
                    if is_optional and not data:
                        if edge_name in self.table:
                            data = [self.table[edge_name][i]]
                        else:
                            data = [pd.NA]
                    else:
                        if pedge.range_:
                            data = [[Edge(e) for e in d] for d in data]
                        else:
                            data = [Edge(d) for d in data]
                    names_to_data[edge_name].append(data)

        for name, data in names_to_data.items():
            self.table[name] = data
        filter_col = []
        for i in range(len(self.table)):
            filter_col.append(all(len(self.table[n][i]) > 0 for n in names_to_data))
        self.table = self.table[filter_col]
        self.table = self.table.explode(list(names_to_data.keys()), ignore_index=True)

        # TODO some kind of pushdown could be implemented here instead
        if where := node.oC_Where():
            self._process_where(where)

    def _process_reading_clause(self, node: CypherParser.OC_ReadingClauseContext):
        assert not node.oC_InQueryCall(), "Unsupported query - CALL not implemented"

        if match_ := node.oC_Match():
            self._process_match(match_)
        elif unwind := node.oC_Unwind():
            self._process_unwind(unwind)

    def _interpret_pattern(
        self, pattern: CypherParser.OC_PatternContext
    ) -> pattern_graph.Graph:
        pgraph = pattern_graph.Graph()

        pattern_part = pattern.oC_PatternPart()
        assert pattern_part

        for part in pattern_part:
            assert (
                not part.oC_Variable()
            ), "Unsupported query - named paths not supported"
            anon_part = part.oC_AnonymousPatternPart()
            assert anon_part
            pgraph.add_fragment(anon_part)
        return pgraph

    def _evaluate_pattern_graph_properties(
        self, pgraph: pattern_graph.Graph
    ) -> Tuple[
        Dict[pattern_graph.NodeID, pd.Series], Dict[pattern_graph.EdgeID, pd.Series]
    ]:
        node_ids_to_props = {}
        for nid, n in pgraph.nodes.items():
            if n.name and n.name in self.table:
                if n.properties or n.labels:
                    raise ExecutionError("SyntaxError::VariableAlreadyBound")
            if n.properties:
                node_ids_to_props[nid] = self._evaluate_map_literal(n.properties)
        edge_ids_to_props = {}
        for eid, e in pgraph.edges.items():
            if e.properties:
                edge_ids_to_props[eid] = self._evaluate_map_literal(e.properties)

        return node_ids_to_props, edge_ids_to_props

    def _process_create(self, node: CypherParser.OC_CreateContext):
        pattern = node.oC_Pattern()
        assert pattern
        pgraph = self._interpret_pattern(pattern)
        node_ids_to_props, edge_ids_to_props = self._evaluate_pattern_graph_properties(
            pgraph
        )

        entities_to_data = {}
        for n in pgraph.nodes.values():
            if n.name:
                assert n.name not in entities_to_data, "Duplicate name"
                entities_to_data[n.name] = []
        for e in pgraph.edges.values():
            if e.name:
                assert e.name not in entities_to_data, "Duplicate name"
                entities_to_data[e.name] = []

        for eid, e in pgraph.edges.items():
            if e.undirected:
                raise ExecutionError("Creating undirected edges not allowed")
            if len(e.types) != 1:
                raise ExecutionError("Created edges must have 1 type")
            if e.range_ is not None:
                raise ExecutionError("Can't create variable length edge")

        for i in range(len(self.table)):
            node_id_to_data_id = {}
            for nid, n in pgraph.nodes.items():
                if n.name and n.name in self.table:
                    assert not n.labels, "Cannot create bound node"
                    assert not n.properties, "Cannot create bound node"
                    data_node = self.table[n.name][i]
                    assert data_node
                    assert isinstance(data_node, Node), "TypeError, expected node"
                    node_id_to_data_id[nid] = data_node.id_
                else:
                    data_id = self._vend_node_id()
                    node_id_to_data_id[nid] = data_id
                    props = node_ids_to_props.get(nid)
                    data = {
                        "labels": n.labels,
                        "properties": props[i] if props is not None else {},
                    }
                    self.graph.add_node(data_id, **data)
                    data_node = Node(data_id)

                if n.name:
                    if (data := entities_to_data.get(n.name)) is not None:
                        data.append(data_node)

            for eid, e in pgraph.edges.items():
                if e.name and e.name in self.table:
                    assert ExecutionError("SyntaxError::Cannot bind edge in create")
                props = edge_ids_to_props.get(eid)
                data = {
                    "type": list(e.types)[0],
                    "properties": props[i] if props is not None else {},
                }
                start = node_id_to_data_id[e.start]
                end = node_id_to_data_id[e.end]
                key = self.graph.add_edge(start, end, **data)
                if e.name:
                    if (data := entities_to_data.get(e.name)) is not None:
                        data.append(Edge((start, end, key)))

        for name, data in entities_to_data.items():
            self.table[name] = data

    def _process_delete(self, node: CypherParser.OC_DeleteContext):
        is_detach = False
        assert node.children
        if node.children[0].getText().lower() == "detach":
            is_detach = True
        nodes_to_delete = set()
        edges_to_delete = set()
        exprs = node.oC_Expression()
        assert exprs is not None
        for expr in exprs:
            output = self._evaluate_expression(expr)
            for entity in output:
                if entity is pd.NA:
                    continue
                if isinstance(entity, Node):
                    nodes_to_delete.add(entity.id_)
                elif isinstance(entity, Edge):
                    edges_to_delete.add(entity.id_)
                else:
                    raise ExecutionError("TypeError::DeleteNonEntity")

        if not is_detach:
            for node in nodes_to_delete:
                for edge in self.graph.out_edges(node, keys=True):
                    if edge not in edges_to_delete:
                        raise ExecutionError("DeleteError::DeleteAttachedNode")
                for edge in self.graph.in_edges(node, keys=True):
                    if edge not in edges_to_delete:
                        raise ExecutionError("DeleteError::DeleteAttachedNode")

        for edge in edges_to_delete:
            self.graph.remove_edge(*edge)
        for node_ in nodes_to_delete:
            self.graph.remove_node(node_)
            self._deleted_ids.add(node_)

    def _set_prop(self, src: Any, path: List[str], value: Any):
        if isinstance(src, Node):
            self._set_prop(self.graph.nodes[src.id_]["properties"], path, value)
        elif isinstance(src, Edge):
            self._set_prop(self.graph.edges[src.id_]["properties"], path, value)
        elif isinstance(src, dict):
            if len(path) == 1:
                src[path[0]] = value
            else:
                self._set_prop(src[path[0]], path[1:], value)
        else:
            raise ExecutionError(f"TypeError - called SET on a {type(src)}")

    def _modify_labels(
        self, node_col, label_exprs: CypherParser.OC_NodeLabelsContext, is_set: bool
    ):
        label_name_exprs = label_exprs.oC_NodeLabel()
        assert label_name_exprs
        labels = set(lexpr.oC_LabelName().getText() for lexpr in label_name_exprs)
        for n in node_col:
            if n is pd.NA:
                continue
            if not isinstance(n, Node):
                raise ExecutionError(
                    "TypeError::Cannot SET/REMOVE label on non-node type"
                )
            if is_set:
                self.graph.nodes[n.id_]["labels"].update(labels)
            else:
                self.graph.nodes[n.id_]["labels"].difference_update(labels)

    def _copy_values(self, dest_col, source_col, update: bool):
        for dst, src in zip(dest_col, source_col):
            source_obj = src
            if isinstance(src, Node):
                source_obj = self.graph.nodes[src.id_]["properties"]
            elif isinstance(src, Edge):
                source_obj = self.graph.edges[src.id_]["properties"]
            elif not isinstance(src, dict):
                raise ExecutionError("Cannot SET properties from a non-map type")

            if isinstance(dst, Node):
                if update:
                    self.graph.nodes[dst.id_]["properties"].update(source_obj)
                else:
                    self.graph.nodes[dst.id_]["properties"] = source_obj.copy()
            elif isinstance(dst, Edge):
                if update:
                    self.graph.edges[dst.id_]["properties"].update(source_obj)
                else:
                    self.graph.edges[dst.id_]["properties"] = source_obj.copy()
            else:
                raise ExecutionError("Cannot SET properties to a non-graphtype")

    def _process_set(self, node: CypherParser.OC_SetContext):
        set_items = node.oC_SetItem()
        assert set_items
        for set_item in set_items:
            expr = set_item.oC_Expression()
            if self._has_aggregation(expr):
                raise ExecutionError("Cannot aggregate in SET")

            if prop_expr := set_item.oC_PropertyExpression():
                new_values = self._evaluate_expression(expr)
                source = self._evaluate_atom(prop_expr.oC_Atom())
                prop_key_names = [
                    prop.oC_PropertyKeyName().getText()
                    for prop in prop_expr.oC_PropertyLookup()
                ]
                for i, src in enumerate(source):
                    self._set_prop(src, prop_key_names, new_values[i])
            else:
                graph_el_expr = set_item.oC_Variable()
                assert graph_el_expr
                graph_el_name = graph_el_expr.getText()
                if graph_el_name not in self.table:
                    raise ExecutionError(
                        f"SyntaxError::UndefinedVariable {graph_el_name}"
                    )
                graph_el_col = self.table[graph_el_name]
                if label_exprs := set_item.oC_NodeLabels():
                    self._modify_labels(graph_el_col, label_exprs, True)
                else:
                    new_values = self._evaluate_expression(expr)
                    assert set_item.children
                    is_update = any(c.getText() == "+=" for c in set_item.children)
                    self._copy_values(graph_el_col, new_values, is_update)

    def _process_remove_item(self, node: CypherParser.OC_RemoveItemContext):
        if var_expr := node.oC_Variable():
            var_name = var_expr.getText()
            if var_name not in self.table:
                raise ExecutionError(f"SyntaxError::UndefinedVariable {var_name}")
            node_col = self.table[var_name]
            labels = node.oC_NodeLabels()
            assert labels
            self._modify_labels(node_col, labels, False)
            return
        prop_expr = node.oC_PropertyExpression()
        assert prop_expr
        node_expr = prop_expr.oC_Atom()
        assert node_expr
        node_col = self._evaluate_atom(node_expr)
        props = prop_expr.oC_PropertyLookup()
        assert props and len(props) == 1
        prop = props[0]
        prop_name_expr = prop.oC_PropertyKeyName()
        prop_name = prop_name_expr.getText()
        for el in node_col:
            if el is pd.NA:
                continue
            elif isinstance(el, Node):
                data = self.graph.nodes[el.id_]
            elif isinstance(el, Edge):
                data = self.graph.edges[el.id_]
            else:
                raise ExecutionError(
                    f"TypeError::REMOVE expected Node/Edge, got {type(el)}"
                )
            if prop_name in data["properties"]:
                del data["properties"][prop_name]

    def _process_remove(self, node: CypherParser.OC_RemoveContext):
        items = node.oC_RemoveItem()
        assert items
        for item in items:
            self._process_remove_item(item)

    def _process_updating_clause(self, node: CypherParser.OC_UpdatingClauseContext):
        if create := node.oC_Create():
            self._process_create(create)
        elif delete := node.oC_Delete():
            self._process_delete(delete)
        elif set_ := node.oC_Set():
            self._process_set(set_)
        elif remove := node.oC_Remove():
            self._process_remove(remove)
        else:
            raise AssertionError(
                "Unsupported query - only CREATE/DELETE updates supported"
            )

    def _process_single_part_query(self, node: CypherParser.OC_SinglePartQueryContext):
        assert node.children
        for child in node.children:
            if isinstance(child, CypherParser.OC_ReadingClauseContext):
                self._process_reading_clause(child)
            if isinstance(child, CypherParser.OC_UpdatingClauseContext):
                self._process_updating_clause(child)
            if isinstance(child, CypherParser.OC_ReturnContext):
                self._process_return(child)
                break

    def _process_multi_part_query(self, node: CypherParser.OC_MultiPartQueryContext):
        assert node.children
        for child in node.children:
            if isinstance(child, CypherParser.OC_ReadingClauseContext):
                self._process_reading_clause(child)
            if isinstance(child, CypherParser.OC_UpdatingClauseContext):
                self._process_updating_clause(child)
            if isinstance(child, CypherParser.OC_WithContext):
                self._process_with(child)
            if isinstance(child, CypherParser.OC_SinglePartQueryContext):
                self._process_single_part_query(child)

    def _getAST(self, query: str, get_root=None):
        error_listener = GeneratorErrorListener()

        input_stream = InputStream(query)
        lexer = CypherLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)

        stream = CommonTokenStream(lexer)

        parser = CypherParser(stream)
        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)

        if get_root is not None:
            root = get_root(parser)
        else:
            root = parser.oC_Cypher()

        if error_listener.errors_caught > 0:
            raise ExecutionError("Failed to parse query")
        return root

    def exec(self, query_str: str) -> pd.DataFrame:
        self.reset_table()
        self._returned = False

        ast = self._getAST(query_str)

        root = ast.oC_Statement()
        assert root

        query = root.oC_Query()
        assert not hasType(
            query, CypherParser.OC_MergeContext
        ), "Unsupported query - merge not implemented"
        assert not hasType(
            query, CypherParser.OC_UnionContext
        ), "Unsupported query - union not implemented"
        assert not query.oC_StandaloneCall(), "Unsupported query - call not implemented"

        regular_query = query.oC_RegularQuery()
        assert regular_query

        single_query = regular_query.oC_SingleQuery()
        if multi_query := single_query.oC_MultiPartQuery():
            self._process_multi_part_query(multi_query)
        else:
            self._process_single_part_query(single_query.oC_SinglePartQuery())

        if not self._returned:
            self.table = pd.DataFrame()

        if " " in self.table:
            del self.table[" "]
        return self.table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="store")
    parser.add_argument("--file", action="store")
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    exe = CypherExecutor()
    if args.interactive:
        import readline

        while query := input("> "):
            try:
                table = exe.exec(query)
                if len(table):
                    print(table)
            except ExecutionError as e:
                print(e, file=sys.stderr)
                print(e.traceback, file=sys.stderr)

    assert args.query or args.file, "One of --query and --file is required!"

    if args.query:
        query = args.query
    else:
        with open(args.file) as f:
            query = f.read()
    table = exe.exec(query)
    print(table)

    # print(exe.table_to_json())
    import os

    if "DEBUG" in os.environ:
        print(exe.graph)
        print(exe.graph.nodes)
        for n in exe.graph.nodes:
            print(n, exe.graph.nodes[n])
        for e in exe.graph.edges:
            print(e, exe.graph.edges[e])


if __name__ == "__main__":
    main()
