import json
import os
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from config import Configuration
from react_with_memory import run_react_agent
from utils import load_jsonl


def process_item_bcp(item, config):
    """Process a single BrowseComp item."""
    question = item["question"]
    answer = item.get("golden_answers", "")
    _id = item.get("id", "")
    try:
        result = asyncio.run(run_react_agent(question, answer, config, use_memory=config.use_memory))
        return {"id": _id, **result}
    except Exception as e:
        print(f"Error processing item {_id}: {e}")
        return {"id": _id, "question": question, "answer": answer, "error": str(e)}


def process_item_gaia(item, config):
    """Process a single GAIA item."""
    question = item["Question"]
    answer = item.get("answer", "")
    _id = item.get("id", "")
    try:
        result = asyncio.run(run_react_agent(question, answer, config, use_memory=config.use_memory))
        return {"id": _id, **result}
    except Exception as e:
        print(f"Error processing item {_id}: {e}")
        return {"id": _id, "question": question, "answer": answer, "error": str(e)}


def process_item_ww(item, config):
    """Process a single WebWalker item with retries."""
    question = item["Question"]
    answer = item.get("answer", "")
    level = item.get("level", "")
    
    for attempt in range(3):
        try:
            result = asyncio.run(run_react_agent(question, answer, config, use_memory=config.use_memory))
            return {"level": level, **result}
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            last_exception = e
    
    return {"question": question, "answer": answer, "level": level, "error": str(last_exception)}


def run_evaluation(config: Configuration):
    """Run evaluation based on the configured task."""
    eval_task = config.eval_task.lower()
    version = config.version

    if eval_task == "bcp":
        eval_data_path = "data/BrowseComp/eval_100.jsonl"
        results_dir = "data/BrowseComp/results"
        os.makedirs(results_dir, exist_ok=True)
        eval_data = load_jsonl(eval_data_path)
        process_func = process_item_bcp
        result_path = os.path.join(results_dir, f"eval_results_{version}_with_memory.jsonl")

    elif eval_task == "gaia":
        eval_data_path = "data/GAIA/dev_long.json"
        results_dir = "data/GAIA/results"
        os.makedirs(results_dir, exist_ok=True)
        with open(eval_data_path, "r") as f:
            eval_data = json.load(f)
        process_func = process_item_gaia
        result_path = os.path.join(results_dir, f"eval_results_{version}_with_memory.jsonl")

    elif eval_task == "ww":
        eval_data_path = "data/webwalker/dev_long.json"
        results_dir = "data/webwalker/results"
        os.makedirs(results_dir, exist_ok=True)
        with open(eval_data_path, "r") as f:
            eval_data = json.load(f)
        process_func = process_item_ww
        result_path = os.path.join(results_dir, f"eval_results_{version}_with_memory.jsonl")

    else:
        raise ValueError(f"Unsupported eval_task: {eval_task}")

    print(f"Running evaluation for task: {eval_task}")
    print(f"Processing {len(eval_data)} items")
    print(f"Results will be saved to: {result_path}")

    with open(result_path, "w") as f:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_func, item, config) for item in eval_data]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    result = future.result()
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"Completed {i}/{len(eval_data)} items")
                except Exception as e:
                    print(f"Error processing future: {e}")

    print(f"Evaluation complete. Results saved to {result_path}")


if __name__ == "__main__":
    """
    Run evaluation tasks with MemoBrain.
    
    This script requires three models to be deployed:
    1. Reasoning Model (default: Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
       - Main agent for complex reasoning tasks
    2. Auxiliary Model (default: Qwen/Qwen2.5-14B-Instruct)
       - For webpage content summarization
    3. Memory Model (default: TommyChien/MemoBrain-14B)
       - For memory management operations
    
    Deploy models with vLLM:
        vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --port 8000
        vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001
        vllm serve TommyChien/MemoBrain-14B --port 8002
    """
    config = Configuration.from_runnable_config()
    print("=" * 60)
    print("ReAct Agent Evaluation Configuration")
    print("=" * 60)
    print(f"  Task: {config.eval_task}")
    print(f"  Version: {config.version}")
    print(f"  Use Memory: {config.use_memory}")
    print(f"  Reasoning Model: {config.reasoning_model}")
    print(f"    URL: {config.reasoning_model_base_url}")
    print(f"  Auxiliary Model: {config.auxiliary_model}")
    print(f"    URL: {config.auxiliary_model_base_url}")
    if config.use_memory:
        print(f"  Memory Model: {config.memory_model}")
        print(f"    URL: {config.memory_model_base_url}")
        print(f"  Max Memory Size: {config.max_memory_size} tokens")
    print(f"  Max LLM Calls: {config.max_llm_call_per_run}")
    print("=" * 60)
    print()
    
    run_evaluation(config)
