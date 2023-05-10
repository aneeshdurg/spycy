#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from antlr4.error.ErrorListener import ErrorListener

from antlr4 import *
from spycy import pattern_graph
from spycy.dfsmatcher import DFSMatcher
from spycy.errors import ExecutionError
from spycy.expression_evaluator import ConcreteExpressionEvaluator, ExpressionEvaluator
from spycy.gen.CypherLexer import CypherLexer
from spycy.gen.CypherParser import CypherParser
from spycy.graph import (
    EdgeType,
    Graph,
    NetworkXEdge,
    NetworkXGraph,
    NetworkXNode,
    NodeType,
)
from spycy.matcher import Matcher, MatchResult, MatchResultSet
from spycy.types import Edge, Node, Path
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
class CypherExecutorBase(Generic[NodeType, EdgeType]):
    graph: Graph[NodeType, EdgeType]
    table: pd.DataFrame = field(
        default_factory=lambda: CypherExecutorBase._default_table()
    )

    expr_eval: type[
        ExpressionEvaluator[NodeType, EdgeType]
    ] = ConcreteExpressionEvaluator[NodeType, EdgeType]
    matcher: type[Matcher[NodeType, EdgeType]] = DFSMatcher[NodeType, EdgeType]

    _returned: bool = False

    _parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _default_table(cls) -> pd.DataFrame:
        return pd.DataFrame([{" ": 0}])

    def evaluate_expr(
        self,
        expr: Union[CypherParser.OC_ExpressionContext, CypherParser.OC_AtomContext],
        evaluating_aggregation: bool = False,
    ) -> pd.Series:
        return self.expr_eval.evaluate(
            self.table,
            self.graph,
            self._parameters,
            self.matcher,
            expr,
            evaluating_aggregation=evaluating_aggregation,
        )

    def set_params(self, parameters: Dict[str, str]):
        evaluated_params = {}

        old_table = self.table
        self.reset_table()
        for name, src in parameters.items():
            ast = self._getAST(src, get_root=lambda parser: parser.oC_Expression())
            assert ast
            col = self.expr_eval.evaluate(
                CypherExecutorBase._default_table(),
                self.graph,
                evaluated_params,
                self.matcher,
                ast,
            )
            assert len(col) == 1
            evaluated_params[name] = col[0]
        self.table = old_table

        self._parameters = evaluated_params

    def reset_table(self):
        self.table = CypherExecutorBase._default_table()

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
                if self.expr_eval.has_aggregation(expr):
                    raise ExecutionError(
                        "SyntaxError::InvalidAggregation - cannot aggregate during ORDER BY"
                    )
                col = self.evaluate_expr(expr)
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
        skip = self.evaluate_expr(expr)
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
        limit = self.evaluate_expr(expr)
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
            if self.expr_eval.has_aggregation(proj):
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
                expr_column = self.evaluate_expr(
                    expr, evaluating_aggregation=(len(aggregations) > 0)
                )
                alias = get_alias(proj)
                output_table[alias] = expr_column
        else:
            group_by_columns = OrderedDict()
            for alias, proj in group_by_keys.items():
                expr = proj.oC_Expression()
                group_by_columns[alias] = self.evaluate_expr(expr)

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

            # Split each table up into subtables where every subtable has a
            # unique key.

            # Collect the row indicies corresponding to each key and store it in
            # keys_to_mask. Store the row index that will correspond to this key
            # in the output table in keys_to_row
            keys_to_row = {}
            keys_to_mask = {}
            for i in range(len(self.table)):
                key = get_key(i)
                if key not in keys_to_row:
                    for alias in group_by_columns:
                        output_keys[alias].append(group_by_columns[alias][i])
                    keys_to_row[key] = output_keys_row_count
                    output_keys_row_count += 1
                    keys_to_mask[key] = [i]
                else:
                    keys_to_mask[key].append(i)

            # Using the masks, extract the rows corresponding to each key in the
            # input tables into one table per key.
            keys_to_subtable = {}
            for key, mask in keys_to_mask.items():
                keys_to_subtable[key] = self.table.loc[self.table.index[mask]]

            # For each table in keys_to_subtable, run all aggregations
            old_table = self.table
            aggregated_columns = {}
            for alias in aggregations:
                aggregated_columns[alias] = [pd.NA] * len(keys_to_row)
            for key, table in keys_to_subtable.items():
                self.table = table
                for alias, proj in aggregations.items():
                    expr = proj.oC_Expression()
                    col = self.evaluate_expr(expr)
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
        if is_distinct:
            self.table.drop_duplicates(inplace=True, ignore_index=False)

    def _process_return(self, node: CypherParser.OC_ReturnContext):
        body = node.oC_ProjectionBody()
        assert body
        self._process_projection_body(body)
        self._returned = True

    def _filter_table(self, filter_expr: CypherParser.OC_ExpressionContext):
        evaluator = self.expr_eval(
            self.table, self.graph, self._parameters, self.matcher
        )
        self.table = evaluator.filter_table(filter_expr)

    def _process_where(self, node: CypherParser.OC_WhereContext):
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
        list_column = self.evaluate_expr(list_expr)
        alias_var = node.oC_Variable()
        assert alias_var
        alias = alias_var.getText()
        self.table[alias] = list_column
        mask = []
        for i in range(len(self.table)):
            value = self.table[alias][i]
            mask.append(value is not pd.NA and len(value) > 0)
        self.table = self.table[mask]
        self.table = self.table.explode(alias, ignore_index=True)

    def _build_intial_match_result_for_row(
        self, row: int, pgraph: pattern_graph.Graph
    ) -> Optional[MatchResult]:
        initial_state = MatchResult()
        for _, node_ in pgraph.nodes.items():
            if node_.name in self.table:
                continue
            value = self.table[node_.name][row]
            if value is pd.NA:
                return None
            else:
                if not isinstance(value, Node):
                    raise ExecutionError("TypeError cannot rebind as node")
                value = value.id_
            initial_state.node_ids_to_data_ids[node_.id_] = value

        for _, edge in pgraph.edges.items():
            if edge.name not in self.table:
                continue
            value = self.table[edge.name][row]
            if value is pd.NA:
                return None
            else:
                if not isinstance(value, Edge):
                    raise ExecutionError("TypeError cannot rebind as edge")
                value = value.id_
            initial_state.edge_ids_to_data_ids[edge.id_] = value
        return initial_state

    def _process_match(self, node: CypherParser.OC_MatchContext):
        assert node.children
        is_optional = node.children[0].getText().lower() == "optional"

        pattern = node.oC_Pattern()
        assert pattern
        pgraph = self.expr_eval.interpret_pattern(pattern)
        node_ids_to_props, edge_ids_to_props = self._evaluate_pattern_graph_properties(
            pgraph
        )

        filter_ = node.oC_Where()

        names_to_data = {}
        for n in pgraph.nodes.values():
            if n.name:
                names_to_data[n.name] = []
        for e in pgraph.edges.values():
            if e.name:
                names_to_data[e.name] = []
        for p in pgraph.paths:
            names_to_data[p] = []

        result_count = []
        for i in range(len(self.table)):
            initial_state = MatchResult()
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
                results = MatchResultSet()
            else:
                results = self.matcher.match(
                    self.graph,
                    self.table.loc[i].to_dict(),
                    self._parameters,
                    filter_,
                    pgraph,
                    i,
                    node_ids_to_props,
                    edge_ids_to_props,
                    initial_state,
                )

            result_count.append(len(results))
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

            for path_name, path in pgraph.paths.items():
                paths = []
                for _ in range(len(results)):
                    paths.append(Path([], []))
                for nid in path.nodes:
                    data = results.node_ids_to_data_ids.get(nid, [])
                    for i, d in enumerate(data):
                        paths[i].nodes.append(d)
                for eid in path.edges:
                    data = results.edge_ids_to_data_ids.get(eid, [])
                    for i, e in enumerate(data):
                        paths[i].edges.append(e)
                names_to_data[path_name].append(paths)

        for name, data in names_to_data.items():
            self.table[name] = data
        filter_col = []
        for i in range(len(self.table)):
            filter_col.append(all(len(self.table[n][i]) > 0 for n in names_to_data))
        self.table = self.table[filter_col]
        if len(names_to_data) > 0:
            self.table = self.table.explode(
                list(names_to_data.keys()), ignore_index=True
            )
        else:
            self.table["!__replication_hack"] = [[0] * n for n in result_count]
            self.table = self.table.explode("!__replication_hack", ignore_index=True)
            mask = [
                not np.isnan(self.table["!__replication_hack"][i])
                for i in range(len(self.table))
            ]
            self.table = self.table[mask]
            del self.table["!__replication_hack"]

    def _process_reading_clause(self, node: CypherParser.OC_ReadingClauseContext):
        assert not node.oC_InQueryCall(), "Unsupported query - CALL not implemented"

        if match_ := node.oC_Match():
            self._process_match(match_)
        elif unwind := node.oC_Unwind():
            self._process_unwind(unwind)

    def _evaluate_pattern_graph_properties(
        self, pgraph: pattern_graph.Graph
    ) -> Tuple[
        Dict[pattern_graph.NodeID, pd.Series], Dict[pattern_graph.EdgeID, pd.Series]
    ]:
        evaluator = self.expr_eval(
            self.table, self.graph, self._parameters, self.matcher
        )
        return evaluator.evaluate_pattern_graph_properties(pgraph)

    def _process_create(self, node: CypherParser.OC_CreateContext):
        pattern = node.oC_Pattern()
        assert pattern
        pgraph = self.expr_eval.interpret_pattern(pattern)
        for nid, n in pgraph.nodes.items():
            if n.name and n.name in self.table:
                if n.properties or n.labels:
                    raise ExecutionError("SyntaxError::VariableAlreadyBound")

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

        for nid, n in pgraph.nodes.items():
            if n.name and n.name in self.table:
                if n.labels:
                    raise ExecutionError(
                        "SyntaxError::VariableAlreadyBound Cannot create bound node"
                    )
                if n.properties:
                    raise ExecutionError(
                        "SyntaxError::VariableAlreadyBound Cannot create bound node"
                    )
                if pgraph.degree(nid) == 0:
                    raise ExecutionError(
                        "SyntaxError::VariableAlreadyBound Cannot create bound node"
                    )

        for i in range(len(self.table)):
            node_id_to_data_id = {}
            for nid, n in pgraph.nodes.items():
                if n.name and n.name in self.table:
                    data_node = self.table[n.name][i]
                    assert data_node
                    assert isinstance(data_node, Node), "TypeError, expected node"
                    node_id_to_data_id[nid] = data_node.id_
                else:
                    props = node_ids_to_props.get(nid)
                    data = {
                        "labels": n.labels,
                        "properties": props[i] if props is not None else {},
                    }
                    data_id = self.graph.add_node(data)
                    node_id_to_data_id[nid] = data_id
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
                eid = self.graph.add_edge(start, end, data)
                if e.name:
                    if (data := entities_to_data.get(e.name)) is not None:
                        data.append(Edge(eid))

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
            output = self.evaluate_expr(expr)
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
                for edge in self.graph.out_edges(node):
                    if edge not in edges_to_delete:
                        raise ExecutionError("DeleteError::DeleteAttachedNode")
                for edge in self.graph.in_edges(node):
                    if edge not in edges_to_delete:
                        raise ExecutionError("DeleteError::DeleteAttachedNode")

        for edge in edges_to_delete:
            self.graph.remove_edge(edge)
        for node_ in nodes_to_delete:
            self.graph.remove_node(node_)

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
            if self.expr_eval.has_aggregation(expr):
                raise ExecutionError("Cannot aggregate in SET")

            if prop_expr := set_item.oC_PropertyExpression():
                new_values = self.evaluate_expr(expr)
                source = self.evaluate_expr(prop_expr.oC_Atom())
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
                    new_values = self.evaluate_expr(expr)
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
        node_col = self.evaluate_expr(node_expr)
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

    def _process_single_query(self, expr: CypherParser.OC_SingleQueryContext):
        if multi_query := expr.oC_MultiPartQuery():
            self._process_multi_part_query(multi_query)
        else:
            self._process_single_part_query(expr.oC_SinglePartQuery())
        if not self._returned:
            self.table = pd.DataFrame()
        else:
            if " " in self.table:
                del self.table[" "]
            no_columns = len(self.table.columns) == 0
            if no_columns:
                raise ExecutionError("SyntaxError::NoVariablesInScope for WITH/RETURN")

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
        assert not query.oC_StandaloneCall(), "Unsupported query - call not implemented"

        regular_query = query.oC_RegularQuery()
        assert regular_query

        single_query = regular_query.oC_SingleQuery()
        self._process_single_query(single_query)

        unions = regular_query.oC_Union()
        if unions:
            if not self._returned:
                raise ExecutionError("SyntaxError::Union requires return")
            found_is_all = False
            found_not_is_all = False
            for union in unions:
                old_table = self.table
                self.reset_table()

                is_all = False
                assert union.children
                if len(union.children) >= 3:
                    if union.children[2].getText().lower() == "all":
                        if found_not_is_all:
                            raise ExecutionError(
                                "SyntaxError::InvalidClauseComposition cannot mix UNION and UNION ALL"
                            )
                        is_all = True
                        found_is_all = True

                if not is_all:
                    if found_is_all:
                        raise ExecutionError(
                            "SyntaxError::InvalidClauseComposition cannot mix UNION and UNION ALL"
                        )
                    found_not_is_all = True

                next_query = union.oC_SingleQuery()
                assert next_query
                self._process_single_query(next_query)
                if not self._returned:
                    raise ExecutionError("SyntaxError::Union requires return")
                if set(self.table.columns) != set(old_table.columns):
                    raise ExecutionError("UnionError::Column names did not match")

                self.table = pd.concat([old_table, self.table], ignore_index=True)
                if not is_all:
                    self.table = self.table.drop_duplicates(ignore_index=True)

        return self.table


@dataclass
class CypherExecutor(CypherExecutorBase[NetworkXNode, NetworkXEdge]):
    graph: Graph[NetworkXNode, NetworkXEdge] = field(default_factory=NetworkXGraph)


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
