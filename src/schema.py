from typing import List, Optional, Literal
from pydantic import BaseModel, Field


NodeKind = Literal[
    "task",         # The original user/main question.
    "subtask",      # Decomposed/reformulated subquestion.
    "evidence",     # Factual answer or conclusion (obtained via reasoning/tool).
    "summary",      # Node synthesizing a set of previous nodes (created via flush).
]

class AddNodeOp(BaseModel):
    """
    Operation: add a new node to the reasoning graph.
    
    Each node represents a reasoning step: a subtask (question to be investigated)
    OR an evidence node (supporting fact/answer), but NOT "summary" (no new summary nodes
    in this operation).
    """
    tmp_id: str = Field(..., description="Temporary id for this node (e.g., 'tmp1').")
    kind: NodeKind = Field(
        ..., 
        description="Node kind: must be 'subtask' or 'evidence'. Do not create 'summary' nodes via this op."
    )
    thought: List[Notes] = Field(
        ..., 
        description="Notes of the new node: for subtask, this is the decomposed/reformulated question; for evidence, this is the factual answer/conclusion for a (sub)task."
    )

class Notes(BaseModel):
    role: str = Field(..., description="Role of the note.")
    content: str = Field(..., description="Content of the note.")

class AddEdgeOp(BaseModel):
    """
    Operation: add a new edge to the reasoning graph.
    
    Connect new or existing nodes in the current reasoning step.
    Only use the following edge kinds:
        - 'decompose': use when breaking down a (sub)task into subtask(s).
        - 'refine': use when making a (sub)task more specific.
        - 'support': use when an 'evidence' node supports/answers a (sub)task.
    src/dst: 
        - If referring to an existing node: use its id (e.g., 2).
        - If referring to a new node in this patch: use the tmp_id (e.g., 'tmp1').
    """
    src: str = Field(..., description="Source node id or tmp_id (e.g., 2, or 'tmp1').")
    dst: str = Field(..., description="Destination node id or tmp_id (e.g., 3, or 'tmp2').")
    # kind: EdgeKind = Field(..., description="Edge kind.")
    rationale: Optional[str] = Field(
        "", 
        description="Optional: Short rationale for this edge's reasoning (may be empty)."
    )

class MemoryPatch(BaseModel):
    """
    Full patch object output by the Reasoning Agent following the new prompt.
    - add_nodes: list of new node(s) to add (all must be 'subtask' or 'evidence' type).
    - add_edges: list of edge objects to connect new and existing nodes.
    - flush_spans: list of flush span operation(s) (optional, usually empty).
    - switch_spine: optional; change current main path anchor to existing node.
    """
    add_nodes: List[AddNodeOp] = Field(
        default_factory=list,
        description="List of new node objects (must be 'subtask' or 'evidence')."
    )
    add_edges: List[AddEdgeOp] = Field(
        default_factory=list,
        description="List of new edge objects."
    )

class FlushAndFoldOp(BaseModel):
    """
    Operation: flush and fold the reasoning graph.
    """
    flush_ops: List[FlushOp] = Field(
        default_factory=list,
        description="List of flush operation objects."
    )
    fold_ops: List[FoldOp] = Field(
        default_factory=list,
        description="List of fold operation objects."
    )

class FoldOp(BaseModel):
    """
    Operation: fold the reasoning graph.
    """
    ids: List[int] = Field(..., description="List of node ids to fold.")
    notes: List[Notes] = Field(..., description="List of notes for the fold operation.")
    rationale: Optional[str] = Field(..., description="Optional: Short rationale for this fold operation (may be empty).")

class FlushOp(BaseModel):
    """
    Operation: flush the reasoning graph.
    """
    id: int = Field(..., description="Node id to flush.")
    rationale: Optional[str] = Field(..., description="Optional: Short rationale for this flush operation (may be empty).")

if __name__ == "__main__":
    print(MemoryPatch.model_json_schema())
