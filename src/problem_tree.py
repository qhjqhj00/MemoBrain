from dataclasses import dataclass
from typing import Dict, List, Literal, Any, Union
import itertools
import json


NodeKind = Literal[
    "task",      
    "subtask",     
    "evidence",    
    "summary"      
]

@dataclass
class ReasoningNode:
    node_id: int
    kind: NodeKind
    thought: str    
    related_turn_ids: List[int]
    active: Union[bool, str] = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "related_turn_ids": self.related_turn_ids,
            "thought": self.thought,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningNode":
        return cls(
            node_id=data["node_id"],
            kind=data["kind"],
            related_turn_ids=data.get("related_turn_ids", []),
            thought=data["thought"],
            active=data.get("active", True),
        )


@dataclass
class Edge:
    src: str
    dst: str
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "rationale": self.rationale,
        }   

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            src=data["src"],
            dst=data["dst"],
            rationale=data.get("rationale", ""),
        )


class ReasoningGraph: 

    def __init__(self):
        self.nodes: Dict[int, ReasoningNode] = {}
        self.edges: List[Edge] = []
        self._id_counter = itertools.count(1)


    def add_node(
        self,
        kind: NodeKind,
        thought: str,
        related_turn_ids: List[int] = None,
    ) -> ReasoningNode:
        node = ReasoningNode(
            node_id=next(self._id_counter),
            kind=kind,
            thought=thought,
            related_turn_ids=related_turn_ids if related_turn_ids is not None else [],
        )
        self.nodes[node.node_id] = node
        return node

    def add_edge(
        self,
        src: int,
        dst: int,
        rationale: str = "",
    ) -> Edge:
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"Unknown node id in edge: {src} -> {dst}")
        edge = Edge(src=src, dst=dst, rationale=rationale)
        self.edges.append(edge)
        return edge

    def fold_nodes(
        self,
        span_node_ids: List[int],
        thought: str,
        rationale: str = "",
    ) -> ReasoningNode:
        related_turn_ids = []
        for nid in span_node_ids:
            if nid in self.nodes.keys():
                related_turn_ids.extend(self.nodes[nid].related_turn_ids)
        related_turn_ids = list(set(related_turn_ids))
        related_turn_ids.sort()

        summary_node = self.add_node(
            kind="summary",
            thought=thought,
            related_turn_ids=related_turn_ids,
        )

        span_set = set(span_node_ids)
        new_edges: List[Edge] = []

        for e in self.edges:
            if e.src in span_set and e.dst in span_set:
                continue
            elif e.src not in span_set and e.dst in span_set:
                new_edges.append(Edge(src=e.src, dst=summary_node.node_id, rationale=rationale))
            elif e.src in span_set and e.dst not in span_set:
                new_edges.append(Edge(src=summary_node.node_id, dst=e.dst, rationale=e.rationale))
            else:
                new_edges.append(e)

        self.edges = new_edges

        for nid in span_set:
            if nid in self.nodes.keys():
                self.nodes[nid].active = False

        return summary_node

    def flush_node(self, node_id: int) -> None:
        if node_id in self.nodes.keys():
            self.nodes[node_id].active = "Flushed"
            return
        raise ValueError(f"Node {node_id} not found in graph")


    def apply_patch(self, patch_json: Dict[str, Any], related_turn_ids: List[int] = None):
        tempid2realid = {}
        for node in patch_json["add_nodes"]:
            real_node_id = self.add_node(
                kind=node["kind"],
                thought=node["thought"],
                related_turn_ids=related_turn_ids if related_turn_ids is not None else [],
            ).node_id
            tempid2realid[node["tmp_id"]] = real_node_id
        
        def extract_numbers_from_string(s: str) -> list:
            """Return a list of integers found in the string s."""
            import re
            return [int(num) for num in re.findall(r'\d+', s)]
        
        for edge in patch_json["add_edges"]:
            src = str(edge["src"])
            dst = str(edge["dst"])

            if src in tempid2realid:
                src = tempid2realid[src]
            else:
                try:
                    src = int(src)
                except ValueError:
                    print(src)
                    src = extract_numbers_from_string(src)[0]
            if dst in tempid2realid:
                dst = tempid2realid[dst]
            else:
                dst = int(dst)

            self.add_edge(
                src=src,
                dst=dst,
                rationale=edge.get("rationale", ""),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningGraph":
        g = cls()
        g.nodes = {int(nid): ReasoningNode.from_dict(nd) for nid, nd in data["nodes"].items()}
        g.edges = [Edge.from_dict(ed) for ed in data["edges"]]

        max_id = max(g.nodes.keys())
        g._id_counter = itertools.count(max_id + 1)
        return g

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_leaf_node_ids(self) -> List[int]:
        out_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge.src in out_degree:
                out_degree[edge.src] += 1
        leaf_ids = [nid for nid, node in self.nodes.items() if out_degree.get(nid, 0) == 0 and node.active]
        return leaf_ids


    def pretty_print(self, mode: str = "full") -> str:
        lines: List[str] = []

        from collections import defaultdict, deque

        children = defaultdict(list)
        in_degrees = {nid: 0 for nid in self.nodes if self.nodes[nid].active}
        for edge in self.edges:
            if self.nodes.get(edge.src, None) and self.nodes.get(edge.dst, None):
                if self.nodes[edge.src].active and self.nodes[edge.dst].active:
                    children[edge.src].append((edge.dst, edge.rationale))
                    in_degrees[edge.dst] += 1
        roots = [nid for nid, deg in in_degrees.items() if deg == 0 and self.nodes[nid].active]
        visited = set()

        def walk(node_id, indent: str):
            if node_id in visited:
                return
            visited.add(node_id)
            n = self.nodes[node_id]
            if isinstance(n.active, str):
                status = n.active
            else:
                status = "Active" if n.active else "Inactive" 
                
            if mode == "full":
                lines.append(f"{indent}- Node {node_id}: [{n.kind}] [{status}] {n.thought}")
            else:
                lines.append(f"{indent}- Node {node_id}: [{n.kind}] [{status}]")

            for dst, rationale in children.get(node_id, []):
                dst_node = self.nodes.get(dst)
                if dst_node and dst not in visited:
                    edge_info = f"--[->] Node {dst} [Rationale: {rationale}]"
                else:
                    continue
                lines.append(f"{indent}    {edge_info}")
                walk(dst, indent + "        ")
        
        if not roots:
            lines.append("No roots (possibly empty graph)")
        else:
            for root in roots:
                walk(root, "")

        shown_nodes = set(visited)
        hidden_nodes = [nid for nid in self.nodes if self.nodes[nid].active and nid not in shown_nodes]
        if hidden_nodes:
            lines.append("\nIsolated active nodes (not connected to roots):")
            for nid in hidden_nodes:
                n = self.nodes[nid]
                mark = []
                mark_str = f" ({', '.join(mark)})" if mark else ""
                lines.append(f"- {nid}{mark_str}: [{n.kind}] {n.thought}")

        return "\n".join(lines)


