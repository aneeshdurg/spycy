# This example cannot be run directly.
# See https://github.com/aneeshdurg/pyllvmpass to run this file as an LLVM opt
# pass

from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple
from enum import Enum

from spycy.graph import Graph
from spycy.spycy import CypherExecutorBase

import llvmcpy.llvm as cllvm


class LLVMType(Enum):
    Function = "Function"
    BasicBlock = "BasicBlock"
    Instruction = "Instruction"


@dataclass
class LLVMNode:
    obj: Any
    obj_type: LLVMType

    def __hash__(self):
        return hash(self.obj)


NodeType = LLVMNode
EdgeType = Tuple[LLVMNode, LLVMNode]


@dataclass
class LLVMGraphNodes(Mapping[NodeType, Any]):
    module: cllvm.Module

    def make_function_node(self, fn):
        return LLVMNode(fn, LLVMType.Function)

    def make_basic_block_node(self, bb):
        return LLVMNode(bb, LLVMType.BasicBlock)

    def make_inst_node(self, inst):
        return LLVMNode(inst, LLVMType.Instruction)

    def __iter__(self):
        def iterator():
            for fn in self.module.iter_functions():
                yield self.make_function_node(fn)
                for bb in fn.iter_basic_blocks():
                    yield self.make_basic_block_node(bb)
                    for inst in bb.iter_instructions():
                        yield self.make_inst_node(inst)
        return iterator()

    def __getitem__(self, n: LLVMNode):
        props = {}
        if n.obj_type != LLVMType.BasicBlock:
            props["str"] = n.obj.print_value_to_string().decode()
        return {
            "labels": [n.obj_type.value],
            "properties": props,
        }

    def __len__(self):
        raise NotImplementedError()


@dataclass
class LLVMGraphEdges(Mapping[EdgeType, Any]):
    def __iter__(self):
        raise NotImplementedError()

    def __getitem__(self, e: EdgeType):
        type_ = None
        if e[0].obj_type == LLVMType.Function:
            assert e[1].obj_type == LLVMType.BasicBlock
            type_ = "has_block"
        elif e[0].obj_type == LLVMType.BasicBlock:
            if e[1].obj_type == LLVMType.BasicBlock:
                type_ = "controlflow"
            elif e[1].obj_type == LLVMType.Instruction:
                type_ = "first"
            else:
                raise AssertionError("Unexpected dst type for BasicBlock edge")
        elif e[0].obj_type == LLVMType.Instruction:
            assert e[1].obj_type == LLVMType.Instruction
            type_ = "next"

        return {"type": [type_], "properties": {}}


@dataclass
class LLVMGraph(Graph[NodeType, EdgeType]):
    """A Graph implementation that maps graph access to an LLVM Module"""

    module: cllvm.Module

    @property
    def nodes(self) -> Mapping[NodeType, Any]:
        return LLVMGraphNodes(self.module)

    @property
    def edges(self) -> Mapping[EdgeType, Any]:
        return LLVMGraphEdges()

    def add_node(self, *_) -> NodeType:
        raise Exception("LLVMGraph is read-only!")

    def add_edge(self, *_) -> EdgeType:
        raise Exception("LLVMGraph is read-only!")

    def out_edges(self, node: NodeType) -> List[EdgeType]:
        if node.obj_type == LLVMType.Function:
            return [(node, LLVMNode(bb, LLVMType.BasicBlock)) for bb in
                    node.obj.iter_basic_blocks()]
        elif node.obj_type == LLVMType.BasicBlock:
            bb = node.obj
            terminator = bb.get_terminator()
            n_succ = terminator.get_num_successors()
            succ_bbs = [(node, LLVMNode(terminator.get_successor(i), LLVMType.BasicBlock)) for i in range(n_succ)]
            first_inst = next(iter(node.obj.iter_instructions()))

            return [(node, LLVMNode(first_inst, LLVMType.Instruction))] + succ_bbs

        else:
            if node.obj.next_instruction:
                return [(node, LLVMNode(node.obj.next_instruction, LLVMType.Instruction))]
            return []

    def in_edges(self, node: NodeType) -> List[EdgeType]:
        if node.obj_type == LLVMType.Function:
            return []
        elif node.obj_type == LLVMType.BasicBlock:
            return [(LLVMNode(node.obj.get_parent(), LLVMType.Function), node)]
        else:
            if node.obj.previous_instruction:
                return [(LLVMNode(node.obj.previous_instruction, LLVMType.Instruction), node)]
            return [(LLVMNode(node.obj.instruction_parent, LLVMType.BasicBlock), node)]

    def remove_node(self, _):
        raise Exception("LLVMGraph is read-only!")

    def remove_edge(self, _):
        raise Exception("LLVMGraph is read-only!")

    def src(self, edge: EdgeType) -> NodeType:
        return edge[0]

    def dst(self, edge: EdgeType) -> NodeType:
        return edge[1]


class LLVMCypherExecutor(CypherExecutorBase[NodeType, EdgeType]):
    """Enable openCypher queries against a LLVMGraph"""

    def __init__(self, module: cllvm.Module):
        graph = LLVMGraph(module)
        super().__init__(graph=graph)


# Entry point for LLVM opt pass
def run_on_module(module: cllvm.Module):
    exe = LLVMCypherExecutor(module)
    print(exe.exec("""
        match (fn: Function)-->(b: BasicBlock)-[*]->(i) WHERE i.str CONTAINS 'alloca' RETURN fn, count(b)
    """))
    return 0
