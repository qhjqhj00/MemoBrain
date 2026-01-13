import time
import asyncio
import copy
from typing import Dict, Literal, Optional

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from memobrain import MemoBrain

from config import Configuration
from state import OverallState
from utils import today_date, call_server_async, num_tokens_from_messages
from prompts import SYSTEM_PROMPT
from tools import execute_tool_call

async def planning_node(state: OverallState, config: RunnableConfig) -> OverallState:
    """Node that calls the LLM for planning/reasoning."""
    configurable = Configuration.from_runnable_config(config)
    memory = state.get("memory")
    
    if time.time() - state["start_time"] > configurable.max_time_seconds:
        return {
            "prediction": "No answer found after timeout",
            "termination": "timeout",
            "status": state["status"] + ["timeout"]
        }

    if state["num_llm_calls_available"] <= 0:
        last_message = state.get("messages", [])
        if last_message and '<answer>' in last_message[-1].get("content", ""):
            prediction = last_message[-1]["content"].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = "No answer found."
            termination = "exceed available llm calls"
        return {
            "prediction": prediction,
            "termination": termination,
            "status": state["status"] + ["exceeded_calls"]
        }

    messages = state.get("messages", [])
    compressed_messages = state.get("compressed_messages", [])
    
    if not messages or len(messages) == 0:
        system_prompt = SYSTEM_PROMPT + today_date()
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["question"]}
        ]
        if memory:
            memory.messages = copy.deepcopy(initial_messages)
        messages_to_add = initial_messages
        all_messages_for_llm = initial_messages
        current_context_size = state.get("current_working_context_size", num_tokens_from_messages(initial_messages))
    else:
        messages_to_add = []
        if compressed_messages:
            all_messages_for_llm = compressed_messages
        else:
            all_messages_for_llm = messages
        current_context_size = state.get("current_working_context_size", num_tokens_from_messages(all_messages_for_llm))
    
    recall_performed = False
    total_recall_time = state.get("total_recall_time", 0.0)
    
    if memory and current_context_size > configurable.max_memory_size:
        recall_start_time = time.time()
        all_messages_for_llm = await memory.recall()
        recall_time = time.time() - recall_start_time
        total_recall_time += recall_time
        current_context_size = num_tokens_from_messages(all_messages_for_llm)
        state["messages"] = all_messages_for_llm
        recall_performed = True
        if current_context_size > configurable.max_memory_size:
            print(f"Warning: Token count after recall ({current_context_size}) still exceeds max_memory_size")

    content = await call_server_async(
        messages=all_messages_for_llm,
        api_base=configurable.reasoning_model_base_url,
        model_name=configurable.reasoning_model,
        api_key=configurable.reasoning_model_api_key,
        llm_generate_cfg=configurable.llm_generate_cfg
    )

    print(f'Round {state["round"] + 1}: {content}')

    if '<tool_response>' in content:
        pos = content.find('<tool_response>')
        content = content[:pos]

    assistant_message = {"role": "assistant", "content": content.strip()}
    messages_to_add.append(assistant_message)

    new_status = state["status"].copy()
    if '<tool_call>' in content and '</tool_call>' in content:
        new_status.append("tool_call")
    elif '<answer>' in content and '</answer>' in content:
        new_status.append("answer")
    else:
        new_status.append("continue")

    all_messages = all_messages_for_llm + [assistant_message]
    token_count = num_tokens_from_messages(all_messages)
    updated_context_size = token_count
    print(f"round: {state['round'] + 1}, token count: {token_count}\n" + "="*100)
    
    return_dict = {
        "messages": messages_to_add,
        "response": content.strip(),
        "status": new_status,
        "round": state["round"] + 1,
        "num_llm_calls_available": state["num_llm_calls_available"] - 1,
        "token_count": token_count,
        "current_working_context_size": updated_context_size,
        "total_recall_time": total_recall_time,
        "total_memorize_time": state.get("total_memorize_time", 0.0)
    }
    
    if memory and recall_performed:
        return_dict["compressed_messages"] = ["__REPLACE__"] + all_messages_for_llm + [assistant_message]
    elif compressed_messages:
        return_dict["compressed_messages"] = [assistant_message]
    
    return return_dict


async def tool_call_node(state: OverallState, config: RunnableConfig) -> OverallState:
    """Node that executes tool calls."""
    configurable = Configuration.from_runnable_config(config)
    memory = state.get("memory")
    content = state["response"]

    if asyncio.iscoroutinefunction(execute_tool_call):
        result = await execute_tool_call(content, configurable, state["web_pages"])
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, execute_tool_call, content, configurable, state["web_pages"])

    if isinstance(result, dict):
        state["web_pages"].update(result["meta"]) if "meta" in result else {}
        result = result["results"]
    tool_response = "<tool_response>\n" + result + "\n</tool_response>"

    new_messages = [{"role": "user", "content": tool_response}]
    last_message = state["messages"][-1]
    
    total_memorize_time = state.get("total_memorize_time", 0.0)
    if memory:
        memorize_start_time = time.time()
        await memory.memorize([last_message, {"role": "user", "content": tool_response}])
        memorize_time = time.time() - memorize_start_time
        total_memorize_time += memorize_time
    
    current_context_size = state.get("current_working_context_size", 0)
    new_message_tokens = num_tokens_from_messages(new_messages)
    updated_context_size = current_context_size + new_message_tokens
    
    compressed_messages = state.get("compressed_messages", [])
    compressed_messages_to_add = []
    if compressed_messages:
        compressed_messages_to_add = new_messages

    return_dict = {
        "messages": new_messages,
        "status": state["status"] + ["tool_response"],
        "current_working_context_size": updated_context_size,
        "total_memorize_time": total_memorize_time,
        "total_recall_time": state.get("total_recall_time", 0.0)
    }
    
    if compressed_messages_to_add:
        return_dict["compressed_messages"] = compressed_messages_to_add
    
    return return_dict


async def check_limits_node(state: OverallState, config: RunnableConfig) -> OverallState:
    """Node that checks token limits and handles accordingly."""
    configurable = Configuration.from_runnable_config(config)

    all_messages = state.get("messages", [])
    token_count = num_tokens_from_messages(all_messages)

    if token_count > configurable.max_tokens:
        print(f"Token quantity exceeds the limit: {token_count} > {configurable.max_tokens}")

        limit_message = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"

        messages = all_messages.copy()
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = limit_message
        else:
            messages.append({"role": "user", "content": limit_message})

        content = await call_server_async(
            messages=messages,
            api_base=configurable.reasoning_model_base_url,
            model_name=configurable.reasoning_model,
            api_key=configurable.reasoning_model_api_key,
            llm_generate_cfg=configurable.llm_generate_cfg
        )

        new_messages = [{"role": "assistant", "content": content.strip()}]
        all_messages_with_new = messages + new_messages
        updated_context_size = num_tokens_from_messages(all_messages_with_new)

        if '<answer>' in content and '</answer>' in content:
            prediction = content.split('<answer>')[1].split('</answer>')[0]
            termination = 'generate an answer as token limit reached'
        else:
            prediction = content.strip()
            termination = 'format error: generate an answer as token limit reached'

        return {
            "messages": new_messages,
            "response": content.strip(),
            "prediction": prediction,
            "termination": termination,
            "status": state["status"] + ["answer"],
            "token_count": token_count,
            "current_working_context_size": updated_context_size,
            "total_memorize_time": state.get("total_memorize_time", 0.0),
            "total_recall_time": state.get("total_recall_time", 0.0)
        }

    return {}


def router_node(state: OverallState) -> Literal["tool_call", "check_limits", "answer", "continue"]:
    """Router node that determines the next step."""
    if not state.get("status"):
        return "continue"

    last_status = state["status"][-1]

    if last_status == "tool_call":
        return "tool_call"
    elif last_status == "answer":
        return "answer"
    elif last_status == "timeout" or last_status == "exceeded_calls":
        return "answer"
    else:
        return "check_limits"


def finalize_node(state: OverallState) -> OverallState:
    """Finalize the result and extract prediction."""
    if '<answer>' in state.get("response", ""):
        prediction = state["response"].split('<answer>')[1].split('</answer>')[0]
        termination = 'answer'
    else:
        prediction = 'No answer found.'
        termination = 'answer not found'
        if state["num_llm_calls_available"] == 0:
            termination = 'exceed available llm calls'

    return {
        "prediction": prediction,
        "termination": termination
    }


def build_graph() -> StateGraph:
    """Build and return the ReAct agent graph."""
    workflow = StateGraph(OverallState)

    workflow.add_node("planning", planning_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("check_limits", check_limits_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("planning")

    workflow.add_conditional_edges(
        "planning",
        router_node,
        {
            "tool_call": "tool_call",
            "check_limits": "check_limits",
            "answer": "finalize",
            "continue": "check_limits"
        }
    )

    def route_after_limits(state: OverallState) -> Literal["planning", "finalize"]:
        if state.get("status", []) and state["status"][-1] == "answer":
            return "finalize"
        return "planning"

    workflow.add_conditional_edges(
        "check_limits",
        route_after_limits,
        {
            "planning": "planning",
            "finalize": "finalize"
        }
    )

    workflow.add_edge("tool_call", "planning")
    workflow.add_edge("finalize", END)

    return workflow.compile()


async def run_react_agent(
    question: str,
    answer: str = "",
    config: Optional[Configuration] = None,
    use_memory: bool = True
) -> Dict:
    """
    Run the ReAct agent with given inputs.
    
    Args:
        question: The question to answer
        answer: The ground truth answer (optional, for evaluation)
        config: Configuration object with model settings
        use_memory: Whether to use MemoBrain for memory management (default: True)
    
    Returns:
        Dictionary containing results and statistics
    """
    if config is None:
        config = Configuration.from_runnable_config()
    
    memory = None
    if use_memory:
        memory = MemoBrain(
            api_key=config.memory_model_api_key,
            base_url=config.memory_model_base_url,
            model_name=config.memory_model,
        )
        memory.init_memory(question)
    
    system_prompt = SYSTEM_PROMPT + today_date()
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    initial_context_size = num_tokens_from_messages(initial_messages)
    
    process_start_time = time.time()
    initial_state: OverallState = {
        "messages": [],
        "compressed_messages": [],
        "response": "",
        "status": [],
        "web_pages": {},
        "question": question,
        "round": 0,
        "num_llm_calls_available": config.max_llm_call_per_run,
        "start_time": process_start_time,
        "prediction": "",
        "termination": "",
        "answer": answer,
        "token_count": 0,
        "current_working_context_size": initial_context_size,
        "total_memorize_time": 0.0,
        "total_recall_time": 0.0,
        "total_process_time": 0.0,
        "memory": memory
    }

    graph = build_graph()

    config_dict = {
        "configurable": {
            "reasoning_model": config.reasoning_model,
            "reasoning_model_base_url": config.reasoning_model_base_url,
            "reasoning_model_api_key": config.reasoning_model_api_key,
            "auxiliary_model": config.auxiliary_model,
            "auxiliary_model_base_url": config.auxiliary_model_base_url,
            "auxiliary_model_api_key": config.auxiliary_model_api_key,
            "llm_generate_cfg": config.llm_generate_cfg,
            "max_tokens": config.max_tokens,
            "max_time_seconds": config.max_time_seconds,
            "max_llm_call_per_run": config.max_llm_call_per_run,
            "max_memory_size": config.max_memory_size,
        },
        "recursion_limit": 500
    }

    final_state = await graph.ainvoke(initial_state, config_dict)
    
    process_end_time = time.time()
    total_process_time = process_end_time - process_start_time

    result = {
        "question": question,
        "answer": answer,
        "messages": final_state.get("messages", []),
        "compressed_messages": final_state.get("compressed_messages", []),
        "prediction": final_state.get("prediction", ""),
        "termination": final_state.get("termination", ""),
        "token_count": final_state.get("token_count", 0),
        "total_memorize_time": final_state.get("total_memorize_time", 0.0),
        "total_recall_time": final_state.get("total_recall_time", 0.0),
        "total_process_time": total_process_time,
    }
    
    if memory:
        result["memory"] = final_state.get("memory").graph.to_dict()

    return result
