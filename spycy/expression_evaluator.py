import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from spycy.errors import ExecutionError
from spycy.functions import function_registry, is_aggregation
from spycy.gen.CypherParser import CypherParser
from spycy.graph import Graph
from spycy.types import Edge, FunctionContext, Node
from spycy.visitor import visitor


class ExpressionEvaluator(metaclass=ABCMeta):
    def __init__(self, table: pd.DataFrame, graph: Graph, parameters: Dict[str, Any]):
        pass

    @classmethod
    @abstractmethod
    def has_aggregation(cls, ctx) -> bool:
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

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        table: pd.DataFrame,
        graph: Graph,
        parameters: Dict[str, Any],
        expr: Any,
        evaluating_aggregation: bool = False,
    ) -> pd.Series:
        """Returns a DataFrame with a single column"""
        pass

    @abstractmethod
    def filter_table(
        self, filter_expr: CypherParser.OC_ExpressionContext
    ) -> pd.DataFrame:
        pass


@dataclass
class ConcreteExpressionEvaluator(ExpressionEvaluator):
    table: pd.DataFrame
    graph: Graph
    parameters: Dict[str, Any]

    # TODO use _table_accesses to speed up CREATE/MATCH
    _table_accesses: int = 0
    _evaluating_aggregation: bool = False

    @classmethod
    def has_aggregation(cls, ctx) -> bool:
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

    def filter_table(
        self, filter_expr: CypherParser.OC_ExpressionContext
    ) -> pd.DataFrame:
        old_columns = self.table.columns
        filter_col = self._evaluate_expression(filter_expr)
        mask = filter_col.fillna(False)
        if len(mask) and mask.dtype.kind != "b":
            raise ExecutionError("TypeError::Expected boolean expression for WHERE")
        new_table = self.table[mask]
        assert new_table is not None
        new_table.reset_index(drop=True, inplace=True)

        if len(new_table) == 0:
            for c in old_columns:
                new_table[c] = []
        return new_table

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

    def _evaluate_literal(
        self, expr: CypherParser.OC_LiteralContext, negate: bool = False
    ) -> pd.Series:
        if negate and expr.oC_NumberLiteral() is None:
            raise ExecutionError("SyntaxError::Cannot negate non-numeric literal")

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
            if number.oC_IntegerLiteral() and "e" not in nstr:
                dtype = "int64"
            if negate:
                nstr = "-" + nstr
            value = eval(nstr)
            if math.isinf(value):
                raise ExecutionError("SyntaxError::FloatingPointOverflow")
        elif expr.StringLiteral():
            value = eval(expr.getText())
            dtype = str

        assert value is not None, "Unsupported literal type"
        data = [value] * (rows if not self._evaluating_aggregation else 1)
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

        is_agg = is_aggregation(fnname)

        params = []
        if param_exprs := expr.oC_Expression():
            for param_expr in param_exprs:
                if is_agg and ConcreteExpressionEvaluator.has_aggregation(param_expr):
                    raise ExecutionError("SyntaxError::NestedAggregation")
                # Temporarily forget if we're evaluating an aggregation while we
                # evaluate the parameters. This is to allow expressions like:
                # sum(x + 1) to find the right length for `1`
                tmp = self._evaluating_aggregation
                self._evaluating_aggregation = False
                params.append(self._evaluate_expression(param_expr))
                self._evaluating_aggregation = tmp
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
            self.table = self.filter_table(where_expr)
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

        if name not in self.parameters:
            raise ExecutionError("ParameterNotFound")

        value = self.parameters[name]
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
            for when_col, then_col in alternatives:
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
                for when_col, then_col in alternatives:
                    if when_col[i] is pd.NA:
                        output.append(then_col[i])
                        found = True
                        break
                if found:
                    continue
            else:
                for when_col, then_col in alternatives:
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
            return pd.Series([], dtype=object)

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

        if negate:
            # We need to push the negative sign down into literal parsing to
            # allow negative numbers that are beyond the positive int64/float64
            # space.
            assert child.children
            if len(child.children) == 1:
                atom = child.oC_Atom()
                assert atom
                assert atom.children
                if literal := atom.oC_Literal():
                    return self._evaluate_literal(literal, negate=True)
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
            if not isinstance(x, str):
                return pd.NA
            rhs_e: str = rhs[i]
            if x is pd.NA or rhs_e is pd.NA:
                return pd.NA
            return x.startswith(rhs_e)

        def endswith(i: int, x: str):
            if not isinstance(x, str):
                return pd.NA
            rhs_e: str = rhs[i]
            if x is pd.NA or rhs_e is pd.NA:
                return pd.NA
            return x.endswith(rhs_e)

        def contains(i: int, x: str):
            if not isinstance(x, str):
                return pd.NA
            rhs_e: str = rhs[i]
            if x is pd.NA or rhs_e is pd.NA:
                return pd.NA
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
            if element is pd.NA:
                return pd.NA

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

    @classmethod
    def evaluate(
        cls,
        table: pd.DataFrame,
        graph: Graph,
        parameters: Dict[str, Any],
        expr: Any,
        evaluating_aggregation: bool = False,
    ) -> pd.Series:
        evaluator = ConcreteExpressionEvaluator(
            table, graph, parameters, _evaluating_aggregation=evaluating_aggregation
        )
        if isinstance(expr, CypherParser.OC_ExpressionContext):
            return evaluator._evaluate_expression(expr)
        if isinstance(expr, CypherParser.OC_AtomContext):
            return evaluator._evaluate_atom(expr)
        if isinstance(expr, CypherParser.OC_MapLiteralContext):
            return evaluator._evaluate_map_literal(expr)

        raise ExecutionError("Unsupported expression type passed to evaluate")
