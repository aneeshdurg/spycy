#!/usr/bin/env python3
import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Tuple

import networkx as nx
import pandas as pd
from antlr4.error.ErrorListener import ErrorListener

import pattern_graph
from antlr4 import *
from functions import function_registry
from gen.CypherLexer import CypherLexer
from gen.CypherParser import CypherParser
from visitor import hasType


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

    @dataclass
    class Node:
        id_: int

    @dataclass
    class Edge:
        id_: Tuple[int, int, int]

    def _evaluate_list_literal(
        self, expr: CypherParser.OC_ListLiteralContext
    ) -> pd.Series:
        rows = len(self.table)
        data = []
        for _ in range(rows):
            data.append([])
        elems = expr.oC_Expression()
        assert elems
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
        if (list_literal := expr.oC_ListLiteral()) :
            return self._evaluate_list_literal(list_literal)

        if (map_literal := expr.oC_MapLiteral()) :
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
                raise Exception("FloatingPointOverflow")
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
        return function_registry(fnname, params, self.table)

    def _evaluate_atom(self, expr: CypherParser.OC_AtomContext) -> pd.Series:
        if (literal := expr.oC_Literal()) :
            return self._evaluate_literal(literal)
        if (parameter := expr.oC_Parameter()) :
            return self._evaluate_parameter(parameter)
        if (case_ := expr.oC_CaseExpression()) :
            return self._evaluate_case(case_)
        if (list_comp := expr.oC_ListComprehension()) :
            return self._evaluate_list_comp(list_comp)
        if (pattern_comp := expr.oC_PatternComprehension()) :
            return self._evaluate_pattern_comp(pattern_comp)
        if (rels := expr.oC_Quantifier()) :
            return self._evaluate_quantifier(rels)
        if (par_expr := expr.oC_ParenthesizedExpression()) :
            return self._evaluate_expression(par_expr.oC_Expression())
        if (func_call := expr.oC_FunctionInvocation()) :
            return self._evaluate_function_invocation(func_call)
        if (existential_subquery := expr.oC_ExistentialSubquery()) :
            return self._evaluate_existential_subquery(existential_subquery)
        if (variable := expr.oC_Variable()) :
            self._table_accesses += 1
            # assert not variable.EscapedSymbolicName(), "Unsupported query - variable in `` unsupported"
            return self.table[variable.getText()]

        assert expr.children
        operation = expr.children[0].getText()
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
            return pd.Series(l[r] for l, r in zip(lhs, rhs))

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
                raise AssertionError("Unsupported query - property lookups unsupported")

            if isinstance(child, CypherParser.OC_NodeLabelsContext):
                raise AssertionError("Unsupported query - lables unsupported")
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
        if op == "=":
            return lhs == rhs
        if op == "<>":
            return lhs != rhs

        return eval(f"lhs {op} rhs")

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
        return pd.Series(lhs[i] in e for i, e in enumerate(rhs))

    def _evaluate_null_predicate(
        self, lhs: pd.Series, expr: CypherParser.OC_NullPredicateExpressionContext
    ) -> pd.Series:
        return lhs.apply(lambda x: x is pd.NA)

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
        # 1 child is the ComparisionExpr, the rest are ["NOT", "SP"]
        not_count = (len(expr.children) - 1) / 2
        negate = not_count % 2 == 1

        comp_expr = expr.oC_ComparisonExpression()
        assert comp_expr
        output = self._evaluate_comparision(comp_expr)
        if negate:
            return ~output
        return output

    def _evaluate_and(self, expr: CypherParser.OC_AndExpressionContext) -> pd.Series:
        expressions = expr.oC_NotExpression()
        assert expressions

        output = self._evaluate_not(expressions[0])
        for and_expr in expressions[1:]:
            output = output & self._evaluate_not(and_expr)
        return output

    def _evaluate_xor(self, expr: CypherParser.OC_XorExpressionContext) -> pd.Series:
        expressions = expr.oC_AndExpression()
        assert expressions

        output = self._evaluate_and(expressions[0])
        for and_expr in expressions[1:]:
            output = output ^ self._evaluate_and(and_expr)
        return output

    def _evaluate_or(self, expr: CypherParser.OC_OrExpressionContext) -> pd.Series:
        expressions = expr.oC_XorExpression()
        assert expressions

        output = self._evaluate_xor(expressions[0])
        for xor_expr in expressions[1:]:
            output = output | self._evaluate_xor(xor_expr)
        return output

    def _evaluate_expression(
        self, expr: CypherParser.OC_ExpressionContext
    ) -> pd.Series:
        """Returns a DataFrame with a single column"""
        or_expr = expr.oC_OrExpression()
        assert or_expr
        return self._evaluate_or(or_expr)

    def _process_projection_body(self, node: CypherParser.OC_ProjectionBodyContext):
        is_distinct = node.DISTINCT()
        assert not is_distinct, "Unsupported query - DISTINCT not implemented"

        proj_items = node.oC_ProjectionItems()
        assert proj_items

        if proj_items.children[0].getText() == "*":
            assert len(proj_items.children) == 1
            return

        output_table = pd.DataFrame()
        for proj in proj_items.oC_ProjectionItem():
            expr = proj.oC_Expression()
            expr_column = self._evaluate_expression(expr)
            var = proj.oC_Variable()
            if var:
                alias = var.getText()
            else:
                alias = expr.getText()
            output_table[alias] = expr_column
        self.table = output_table

        assert not node.oC_Order(), "Unsupported query - ORDER BY not implemented"
        assert not node.oC_Skip(), "Unsupported query - ORDER BY not implemented"
        assert not node.oC_Limit(), "Unsupported query - ORDER BY not implemented"

    def _process_return(self, node: CypherParser.OC_ReturnContext):
        body = node.oC_ProjectionBody()
        assert body
        self._process_projection_body(body)
        self._returned = True

    def _process_where(self, node: CypherParser.OC_WhereContext):
        filter_expr = node.oC_Expression()
        assert filter_expr
        filter_col = self._evaluate_expression(filter_expr)
        new_table = self.table[filter_col]
        assert new_table is not None
        self.table = new_table

    def _process_with(self, node: CypherParser.OC_WithContext):
        body = node.oC_ProjectionBody()
        assert body
        self._process_projection_body(body)

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

    def _process_reading_clause(self, node: CypherParser.OC_ReadingClauseContext):
        assert not node.oC_Match(), "Unsupported query - MATCH not implemented"
        assert not node.oC_InQueryCall(), "Unsupported query - CALL not implemented"

        if unwind := node.oC_Unwind():
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

    def _process_create(self, node: CypherParser.OC_CreateContext):
        pattern = node.oC_Pattern()
        assert pattern
        pgraph = self._interpret_pattern(pattern)
        node_ids_to_props = {}
        for nid, n in pgraph.nodes.items():
            if n.properties:
                node_ids_to_props[nid] = self._evaluate_map_literal(n.properties)
        edge_ids_to_props = {}
        for eid, e in pgraph.edges.items():
            if e.properties:
                edge_ids_to_props[eid] = self._evaluate_map_literal(e.properties)

        entities_to_data = {}
        for n in pgraph.nodes.values():
            if n.name:
                assert n.name not in entities_to_data, "Duplicate name"
                entities_to_data[n.name]  = []
        for e in pgraph.edges.values():
            if e.name:
                assert e.name not in entities_to_data, "Duplicate name"
                entities_to_data[e.name]  = []

        for i in range(len(self.table)):
            node_id_to_data_id = {}
            for nid, n in pgraph.nodes.items():
                data_id = len(self.graph.nodes)
                node_id_to_data_id[nid] = data_id
                props = node_ids_to_props.get(nid)
                data = {
                    "labels": n.labels,
                    "properties": props[i] if props is not None else {},
                }
                self.graph.add_node(data_id, **data)
                if n.name:
                    if (data := entities_to_data.get(n.name)) is not None:
                        data.append(CypherExecutor.Node(data_id))

            for eid, e in pgraph.edges.items():
                assert not e.undirected, "Creating undirected edges not allowed"
                assert len(e.types) == 1, "Created edges must have 1 type"
                assert e.range_ is None, "Can't create variable length edge"
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
                        data.append(CypherExecutor.Edge((start, end, key)))

        for name, data in entities_to_data.items():
            self.table[name] = data

    def _process_updating_clause(self, node: CypherParser.OC_UpdatingClauseContext):
        create = node.oC_Create()
        assert create, "Unsupported query - only CREATE updates are allowed"
        self._process_create(create)

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

    def _getAST(self, query: str):
        error_listener = GeneratorErrorListener()

        input_stream = InputStream(query)
        lexer = CypherLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)

        stream = CommonTokenStream(lexer)

        parser = CypherParser(stream)
        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)

        root = parser.oC_Cypher()

        if error_listener.errors_caught > 0:
            raise Exception("Failed to parse query")
        return root

    def exec(self, query_str: str) -> pd.DataFrame:
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
            table = exe.exec(query)
            print(table)

    assert args.query or args.file, "One of --query and --file is required!"

    if args.query:
        query = args.query
    else:
        with open(args.file) as f:
            query = f.read()
    table = exe.exec(query)
    print(table)
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
