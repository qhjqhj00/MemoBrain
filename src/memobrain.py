from openai import AsyncOpenAI
from typing import List, Dict, Any
import json
from .prompts import *
from .problem_tree import ReasoningGraph
import asyncio

class MemoBrain:
    def __init__(self, 
        api_key: str,
        base_url: str,
        model_name: str,
        ):
        self.graph = ReasoningGraph()
        self.messages = []
        self.model_name = model_name
        
        self.agent = AsyncOpenAI(
            api_key=api_key, base_url=base_url)

    def init_memory(self, task):
        self.graph.add_node(
            kind="task", 
            thought=f"Begin to solve the task: {task}", 
            related_turn_ids=[0,1])

    async def memorize(self, new_messages: List[Dict]):
        start_idx = len(self.messages) 
        self.messages.extend(new_messages)
        
        grouped = self._group_pairs(start_idx)
        print(f"{len(grouped)} pairs to memorize...")
        for pair in grouped:
            patch_json = await self._generate_patch(pair)
            try:
                self.graph.apply_patch(patch_json, [start_idx, start_idx+1])
            except:
                print("apply patch failed, continue...")
                continue
            start_idx += 2

    def _group_pairs(self, start_idx: int):
        grouped = []
        temp = []
        for msg in self.messages[start_idx:]:
            if msg["role"] in ("user", "assistant"):
                temp.append(msg)
                if len(temp) == 2:
                    grouped.append(temp)
                    temp = []
        return grouped

    async def _generate_patch(self, pair):
        round_info = json.dumps(pair, ensure_ascii=False)
        graph_str = self.graph.pretty_print() 

        current_message = f"CURRENT_INTERACTION:\n{round_info}\n\n{graph_str}"

        messages = [
            {"role": "system", "content": MEMORY_SYS_PROMPT},
            {"role": "user", "content": current_message}
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._create_completion(messages, stream=False)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"generate patch failed: {e}, retry in 3 seconds ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(3)
                else:
                    print("generate patch failed, raise exception.")
                    raise

        patch_json = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": response.choices[0].message.content})

        try:
            patch_json = json.loads(response.choices[0].message.content)
        except Exception as e:
            print("JSON decode error:", e)
            print("Raw content:", response.choices[0].message.content)
            raise e
        return patch_json
    
    async def _create_completion(self, messages, stream=False):

        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=8*1024,
            top_p=0.8,
            presence_penalty=1.1,
            extra_body={
                "top_k": 20,
                "min_p": 0.05,
            },
        )
        if stream:
            response_content = ""
            async for chunk in await self.agent.chat.completions.create(**kwargs, stream=True):
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    print(delta, end='', flush=True)
                    response_content += delta
            class FakeChoice:
                def __init__(self, content):
                    self.message = type('msg', (), {})()
                    self.message.content = content
            class FakeResponse:
                def __init__(self, content):
                    self.choices = [FakeChoice(content)]
            print() 
            return FakeResponse(response_content)
        else:
            return await self.agent.chat.completions.create(**kwargs)

    async def recall(self):
        graph_str = self.graph.pretty_print()
        messages = [
            {"role": "system", "content": FLUSH_AND_FOLD_PROMPT},
            {"role": "user", "content": f"CURRENT_GRAPH:\n{graph_str}"}
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._create_completion(messages, stream=False)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"recall failed: {e}, retry in 3 seconds ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(3)
                else:
                    print("recall failed, raise exception.")
                    raise

        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
        except Exception as e:
            print("Error parsing recall response as JSON:", e)
            print("Raw recall response:", response.choices[0].message.content)
            raise e
        for flush_op in result["flush_ops"]:
            self.graph.flush_node(flush_op["id"])

        for fold_op in result["fold_ops"]:
            self.graph.fold_nodes(fold_op["ids"], json.dumps(fold_op["notes"]), fold_op["rationale"])

        return self._organize()

    
    def _organize(self):
        
        ops_list = []
        active_ids = []
        summary_dict = {}
        total_messages = len(self.messages)
        protected_indices = set()
        protected_indices.update(range(min(3, total_messages)))
        if total_messages > 3:
            protected_indices.update(range(max(3, total_messages - 4), total_messages))

        for node in self.graph.nodes.values():
            is_active = True if node.active is True else False
            kind = node.kind
            related_turn_ids = node.related_turn_ids
            thought = node.thought

            if kind == "summary":
                if related_turn_ids:
                    last_tid = max(related_turn_ids)
                    summary_dict[last_tid] = json.loads(thought) if isinstance(thought, str) else thought
            elif not is_active:
                if isinstance(thought, str):
                    try:
                        t = json.loads(thought)
                    except:
                        t = []
                else:
                    t = thought
                for idx, tid in enumerate(related_turn_ids):
                    if t and len(t) > idx:
                        ops_list.append({"turn_id": tid, "new_message": t[idx]})
            elif is_active and kind:
                for tid in related_turn_ids:
                    active_ids.append(tid)

        summary_turn_ids_to_remove = set()
        for node in self.graph.nodes.values():
            if node.kind == "summary":
                for tid in node.related_turn_ids:
                    if tid not in protected_indices:
                        summary_turn_ids_to_remove.add(tid)

        replace_dict = {}
        for change in ops_list:
            tid = change["turn_id"]
            if tid not in active_ids and tid not in protected_indices and tid not in summary_turn_ids_to_remove:
                replace_dict[tid] = change["new_message"]

        result_messages = []

        for idx, original_msg in enumerate(self.messages):
            if idx in summary_dict and idx not in protected_indices:
                summary_msgs = summary_dict[idx]
                result_messages.extend(summary_msgs)
                
            if idx in summary_turn_ids_to_remove:
                continue
            
            if idx in replace_dict:
       
                result_messages.append(replace_dict[idx])
            else:
                result_messages.append(original_msg)
            

        return result_messages


    def save_memory(self, file_path: str):
        graph_dict = self.graph.to_dict()
        with open(file_path, "w") as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=4)
        
    
    def load_memory(self, file_path: str):
        with open(file_path, "r") as f:
            json_dict = json.load(f)
        self.graph = ReasoningGraph.from_dict(json_dict)

    def load_dict_memory(self, graph_dict: Dict):
        self.graph = ReasoningGraph.from_dict(graph_dict)
    
