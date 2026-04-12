"""Predicate pushdown for WHERE clauses on MATCH patterns.

When a query says::

    MATCH (a)<-[*1..30]-(b)
    WHERE a.path = '/foo' AND b.path = '/bar'
    RETURN ...

the query semantically constrains both endpoints of the MATCH pattern,
but the executor would otherwise enumerate every structural candidate
across the whole graph and only filter via WHERE afterwards. On a
316-node graph the DFS produces ~6 700 candidate rows, the WHERE filter
keeps one. The result is correct but ~600x slower than necessary.

This module recognises ``<var>.<prop> = <const>`` conjunctions in the
WHERE AST and folds them into the pattern node's property dict before
the DFS runs, so :meth:`spycy.dfsmatcher.DFSMatcher.node_matches` can
prune wrong starting candidates immediately.

The supported shape is intentionally narrow:

- Top-level WHERE expression must be a chain of ``AND`` conjunctions
  (no ``OR``, no ``NOT``).
- Each conjunct must be ``<variable>.<property> = <literal>`` or the
  symmetric ``<literal> = <variable>.<property>``.
- The literal may be a string, integer, float, or boolean.
  Null literals are left to the post-DFS evaluator (Cypher null
  equality semantics require three-valued logic).

Anything more complex is left untouched and falls through to the
existing post-DFS WHERE evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from spycy import pattern_graph
from spycy.gen.CypherParser import CypherParser


PushdownTriple = Tuple[str, str, Any]


def collect_pushdown_predicates(
    where_ast: Optional[CypherParser.OC_WhereContext],
) -> List[PushdownTriple]:
    """Walk a WHERE AST and return ``(var, prop, value)`` triples that
    can be folded into pattern node properties.

    Returns an empty list when the WHERE shape is not pushdown-friendly,
    so the caller can pass any WHERE without checking first.
    """
    if where_ast is None:
        return []
    expr = where_ast.oC_Expression()
    if expr is None:
        return []

    or_expr = expr.oC_OrExpression()
    if _has_multiple_rule_children(or_expr):
        return []  # OR present — semantics differ
    xor_expr = _first_rule_child(or_expr)
    if xor_expr is None or _has_multiple_rule_children(xor_expr):
        return []  # XOR present
    and_expr = _first_rule_child(xor_expr)
    if and_expr is None:
        return []

    # Each AND conjunct is examined independently. Conjuncts whose shape
    # is not pushable (OR, NOT, cross-variable, non-literal RHS, etc.)
    # are simply skipped — they remain in the WHERE AST and the post-DFS
    # evaluator handles them. This means the returned list may be a
    # *partial* extraction: some conjuncts pushed, others not.
    triples: List[PushdownTriple] = []
    for child in and_expr.children:
        if not hasattr(child, 'getRuleIndex'):
            continue  # the literal 'AND' tokens
        not_expr = child
        if _has_not_keyword(not_expr):
            continue  # negated terms can't be pushed down naively
        if not hasattr(not_expr, 'oC_ComparisonExpression'):
            continue
        comp = not_expr.oC_ComparisonExpression()
        if comp is None:
            continue
        triple = _try_extract_equality(comp)
        if triple is not None:
            triples.append(triple)
    return triples


def apply_pushdown(
    triples: List[PushdownTriple],
    pgraph: pattern_graph.Graph,
    node_ids_to_props: Dict[pattern_graph.NodeID, pd.Series],
    table_len: int,
) -> None:
    """Fold *triples* into *pgraph*'s node property dicts in place.

    Mutates ``node_ids_to_props`` so that the property check inside
    :meth:`DFSMatcher.node_matches` will see the pushed constraints.
    The matcher gates on ``pnode.id_ in node_ids_to_props``, so adding
    an entry here is sufficient — no mutation of pnode.properties needed.
    """
    if not triples:
        return

    # Group triples by variable name so each pattern node is touched once.
    by_var: Dict[str, Dict[str, Any]] = {}
    for var, prop, value in triples:
        by_var.setdefault(var, {})[prop] = value

    name_to_pnode = {n.name: (nid, n) for nid, n in pgraph.nodes.items() if n.name}

    for var, props in by_var.items():
        target = name_to_pnode.get(var)
        if target is None:
            continue  # WHERE refers to a name that's not a pattern node
        nid, _pnode = target

        existing = node_ids_to_props.get(nid)
        merged_rows: List[Dict[str, Any]] = []
        if existing is None:
            merged_rows = [dict(props) for _ in range(table_len)]
        else:
            for row_value in existing:
                if isinstance(row_value, dict):
                    merged_rows.append({**row_value, **props})
                else:
                    merged_rows.append(dict(props))

        node_ids_to_props[nid] = pd.Series(merged_rows)


# ----- AST navigation helpers ------------------------------------------------

def _has_multiple_rule_children(node) -> bool:
    return sum(1 for c in node.children if hasattr(c, 'getRuleIndex')) > 1


def _first_rule_child(node):
    for c in node.children:
        if hasattr(c, 'getRuleIndex'):
            return c
    return None


def _has_not_keyword(not_expr) -> bool:
    for c in not_expr.children:
        if not hasattr(c, 'getRuleIndex') and c.getText().lower() == 'not':
            return True
    return False


def _drill_to_operand(node):
    """Walk through single-child rules until we reach the operand layer
    (``oC_NonArithmeticOperatorExpression``) or run out of children.
    """
    while node is not None and hasattr(node, 'children') and node.children:
        if 'NonArithmeticOperator' in type(node).__name__:
            return node
        if len(node.children) == 1:
            node = node.children[0]
        else:
            return node
    return node


def _try_extract_equality(
    comparison_expr,
) -> Optional[PushdownTriple]:
    """Recognise ``<var>.<prop> = <literal>`` (in either order)."""
    if not hasattr(comparison_expr, 'children') or len(comparison_expr.children) < 2:
        return None
    lhs = comparison_expr.children[0]

    partial = None
    for c in comparison_expr.children[1:]:
        if hasattr(c, 'getRuleIndex'):
            partial = c
            break
    if partial is None or len(partial.children) < 2:
        return None
    op = partial.children[0].getText()
    if op != '=':
        return None
    rhs = partial.children[-1]

    lhs_d = _drill_to_operand(lhs)
    rhs_d = _drill_to_operand(rhs)

    var_prop = _try_extract_var_prop(lhs_d)
    if var_prop is not None:
        value = _try_extract_literal(rhs_d)
    else:
        var_prop = _try_extract_var_prop(rhs_d)
        value = _try_extract_literal(lhs_d)

    if var_prop is None or value is None:
        return None
    return (var_prop[0], var_prop[1], value)


def _try_extract_var_prop(non_arith_expr) -> Optional[Tuple[str, str]]:
    """Recognise an oC_NonArithmeticOperatorExpression of shape
    ``Atom + PropertyLookup`` and return ``(var_name, prop_name)``.
    """
    if non_arith_expr is None:
        return None
    if not hasattr(non_arith_expr, 'children') or non_arith_expr.children is None:
        return None
    if len(non_arith_expr.children) != 2:
        return None
    atom, prop_lookup = non_arith_expr.children

    if not hasattr(atom, 'oC_Variable'):
        return None
    var_node = atom.oC_Variable()
    if var_node is None:
        return None

    if not hasattr(prop_lookup, 'oC_PropertyKeyName'):
        return None
    key = prop_lookup.oC_PropertyKeyName()
    if key is None:
        return None

    return (var_node.getText(), key.getText())


def _try_extract_literal(non_arith_expr) -> Optional[Any]:
    """Recognise an oC_NonArithmeticOperatorExpression containing a
    single literal atom.  Supports string, integer, float, boolean,
    and null literals.
    """
    if non_arith_expr is None:
        return None
    if not hasattr(non_arith_expr, 'children') or non_arith_expr.children is None:
        return None
    if len(non_arith_expr.children) != 1:
        return None
    atom = non_arith_expr.children[0]
    if not hasattr(atom, 'oC_Literal'):
        return None
    lit = atom.oC_Literal()
    if lit is None:
        return None

    # String literal
    if lit.StringLiteral() is not None:
        text = lit.getText()
        if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
            body = text[1:-1]
            return (body
                    .replace('\\\\', '\x00')
                    .replace("\\'", "'")
                    .replace('\\"', '"')
                    .replace('\\n', '\n')
                    .replace('\\r', '\r')
                    .replace('\\t', '\t')
                    .replace('\\b', '\b')
                    .replace('\\f', '\f')
                    .replace('\x00', '\\'))

    # Boolean literal
    if lit.oC_BooleanLiteral() is not None:
        return lit.getText().lower() == 'true'

    # Null literal — do not push down; Cypher null-equality semantics
    # (null = null → null, not true) must be handled by the post-DFS
    # WHERE evaluator, not by the dict-based properties_match.
    if lit.NULL() is not None:
        return None

    # Numeric literal (integer or float)
    num = lit.oC_NumberLiteral()
    if num is not None:
        text = num.getText()
        if num.oC_IntegerLiteral() is not None:
            return int(text)
        if num.oC_DoubleLiteral() is not None:
            return float(text)

    return None
