from CypherParser import CypherParser

all_ctxs = [
    'OC_AddOrSubtractExpressionContext',
    'OC_AndExpressionContext',
    'OC_AnonymousPatternPartContext',
    'OC_AtomContext',
    'OC_BooleanLiteralContext',
    'OC_CaseAlternativesContext',
    'OC_CaseExpressionContext',
    'OC_ComparisonExpressionContext',
    'OC_CreateContext',
    'OC_CypherContext',
    'OC_DashContext',
    'OC_DeleteContext',
    'OC_DoubleLiteralContext',
    'OC_ExplicitProcedureInvocationContext',
    'OC_ExpressionContext',
    'OC_FilterExpressionContext',
    'OC_FunctionInvocationContext',
    'OC_FunctionNameContext',
    'OC_IdInCollContext',
    'OC_ImplicitProcedureInvocationContext',
    'OC_InQueryCallContext',
    'OC_IntegerLiteralContext',
    'OC_LabelNameContext',
    'OC_LeftArrowHeadContext',
    'OC_LimitContext',
    'OC_ListComprehensionContext',
    'OC_ListLiteralContext',
    'OC_ListOperatorExpressionContext',
    'OC_LiteralContext',
    'OC_MapLiteralContext',
    'OC_MatchContext',
    'OC_MergeActionContext',
    'OC_MergeContext',
    'OC_MultiPartQueryContext',
    'OC_MultiplyDivideModuloExpressionContext',
    'OC_NamespaceContext',
    'OC_NodeLabelContext',
    'OC_NodeLabelsContext',
    'OC_NodePatternContext',
    'OC_NotExpressionContext',
    'OC_NullOperatorExpressionContext',
    'OC_NumberLiteralContext',
    'OC_OrExpressionContext',
    'OC_OrderContext',
    'OC_ParameterContext',
    'OC_ParenthesizedExpressionContext',
    'OC_PartialComparisonExpressionContext',
    'OC_PatternComprehensionContext',
    'OC_PatternContext',
    'OC_PatternElementChainContext',
    'OC_PatternElementContext',
    'OC_PatternPartContext',
    'OC_PowerOfExpressionContext',
    'OC_ProcedureNameContext',
    'OC_ProcedureResultFieldContext',
    'OC_ProjectionBodyContext',
    'OC_ProjectionItemContext',
    'OC_ProjectionItemsContext',
    'OC_PropertiesContext',
    'OC_PropertyExpressionContext',
    'OC_PropertyKeyNameContext',
    'OC_PropertyLookupContext',
    'OC_PropertyOrLabelsExpressionContext',
    'OC_QueryContext',
    'OC_RangeLiteralContext',
    'OC_ReadingClauseContext',
    'OC_RegularQueryContext',
    'OC_RelTypeNameContext',
    'OC_RelationshipDetailContext',
    'OC_RelationshipPatternContext',
    'OC_RelationshipTypesContext',
    'OC_RelationshipsPatternContext',
    'OC_RemoveContext',
    'OC_RemoveItemContext',
    'OC_ReservedWordContext',
    'OC_ReturnContext',
    'OC_RightArrowHeadContext',
    'OC_SchemaNameContext',
    'OC_SetContext',
    'OC_SetItemContext',
    'OC_SinglePartQueryContext',
    'OC_SingleQueryContext',
    'OC_SkipContext',
    'OC_SortItemContext',
    'OC_StandaloneCallContext',
    'OC_StatementContext',
    'OC_StringListNullOperatorExpressionContext',
    'OC_StringOperatorExpressionContext',
    'OC_SymbolicNameContext',
    'OC_UnaryAddOrSubtractExpressionContext',
    'OC_UnionContext',
    'OC_UnwindContext',
    'OC_UpdatingClauseContext',
    'OC_VariableContext',
    'OC_WhereContext',
    'OC_WithContext',
    'OC_XorExpressionContext',
    'OC_YieldItemContext',
    'OC_YieldItemsContext'
]

print("CREATE ")
node_defs = []
for ctx in all_ctxs:
    type_ = ctx.replace("OC_", "").replace("Context", "")
    node_defs.append(f"({ctx} :{type_})");
print ("", ",\n ".join(node_defs))

first = True
edges = ""
for ctx in all_ctxs:
    children = [c for c in dir(eval(f"CypherParser.{ctx}")) if c.startswith('oC')]
    for child in children:
        child = child.replace('oC', "OC") + "Context"
        if not first:
            edges += ",\n"
        first = False
        edges += f"  ({ctx})-[:CHILD]->({child})"
print(f"CREATE\n{edges}")

