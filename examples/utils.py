import time
import concurrent.futures
from pydantic import BaseModel
from typing import List, Dict
import datetime
import re
import random
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, AsyncOpenAI

import json
import tiktoken

encoding = tiktoken.get_encoding("o200k_base")

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

async def call_server_async(
    messages: List[Dict],
    port: int,
    model_name: str,
    api_key: str = "EMPTY",
    llm_generate_cfg: Dict = None,
    max_tries: int = 3,
    stop=["\n<tool_response>", "<tool_response>"],
    stream=False,
    schema: BaseModel = None,
    max_tokens: int = 8196,
    top_p: float = 0.95,
    temperature: float = 0.6,
    presence_penalty: float = 1.1,
    min_p: float = 0.05,
    top_k: int = 20
) -> str:
    """
    Call the LLM server with retry logic.
    
    Args:
        messages: List of messages for the conversation
        api_base: Base URL for the API
        model_name: Name of the model to use
        api_key: API key (default: "EMPTY")
        llm_generate_cfg: LLM generation configuration dict
        max_tries: Maximum number of retry attempts
        stop: Stop sequences
        stream: Whether to stream the response
        schema: Pydantic schema for guided JSON generation
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        temperature: Temperature for sampling
        repetition_penalty: Repetition penalty
        min_p: Minimum probability threshold
        top_k: Top-k sampling parameter
        
    Returns:
        Generated response content as string
    """
    if llm_generate_cfg is None:
        llm_generate_cfg = {}
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=f"http://localhost:{port}/v1",
        timeout=600.0,
    )

    base_sleep_time = 1
    for attempt in range(max_tries):
        try:
            print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
            
            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "top_p": llm_generate_cfg.get('top_p', top_p),
                "temperature": llm_generate_cfg.get('temperature', temperature),
                "stream": stream,
                "stop": stop,
            }
            
            # Add presence_penalty if in config
            request_params["presence_penalty"] = llm_generate_cfg.get('presence_penalty', 1.1)
            
            # Add extra_body for advanced parameters
            extra_body = {
                "min_p": min_p,
                'include_stop_str_in_output': True,
                'top_k': top_k,
            }
            
            if schema:
                extra_body["guided_json"] = schema.model_json_schema()
            
            request_params["extra_body"] = extra_body
            
            response = await client.chat.completions.create(**request_params)
            
            if stream:
                response_content = ""   
                for chunk in response:
                    response_content += chunk.choices[0].message.content
                    print(chunk.choices[0].message.content, end="", flush=True)
                content = response_content
            else:
                content = response.choices[0].message.content

            if content and content.strip():
                print("--- Service call successful, received a valid response ---")
                return content.strip()
            else:
                print(f"Warning: Attempt {attempt + 1} received an empty response.")

        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
        except Exception as e:
            print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

        if attempt < max_tries - 1:
            sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
            sleep_time = min(sleep_time, 1)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print("Error: All retry attempts have been exhausted. The call has failed.")

    return "vllm server error!!!"

def call_server(
    messages: List[Dict],
    port: int,
    model_name: str,
    api_key: str = "EMPTY",
    llm_generate_cfg: Dict = None,
    max_tries: int = 3,
    stop=["\n<tool_response>", "<tool_response>"],
    stream=False,
    schema: BaseModel = None,
    max_tokens: int = 8196,
    top_p: float = 0.95,
    temperature: float = 0.6,
    presence_penalty: float = 1.1,
    min_p: float = 0.05,
    top_k: int = 20
) -> str:
    """
    Call the LLM server with retry logic.
    
    Args:
        messages: List of messages for the conversation
        api_base: Base URL for the API
        model_name: Name of the model to use
        api_key: API key (default: "EMPTY")
        llm_generate_cfg: LLM generation configuration dict
        max_tries: Maximum number of retry attempts
        stop: Stop sequences
        stream: Whether to stream the response
        schema: Pydantic schema for guided JSON generation
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        temperature: Temperature for sampling
        repetition_penalty: Repetition penalty
        min_p: Minimum probability threshold
        top_k: Top-k sampling parameter
        
    Returns:
        Generated response content as string
    """
    if llm_generate_cfg is None:
        llm_generate_cfg = {}
    
    client = OpenAI(
        api_key=api_key,
        base_url=f"http://localhost:{port}/v1",
        timeout=600.0,
    )

    base_sleep_time = 1
    for attempt in range(max_tries):
        try:
            print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
            
            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "top_p": llm_generate_cfg.get('top_p', top_p),
                "temperature": llm_generate_cfg.get('temperature', temperature),
                "stream": stream,
                "stop": stop,
            }
            
            # Add presence_penalty if in config
            request_params["presence_penalty"] = llm_generate_cfg.get('presence_penalty', 1.1)
            
            # Add extra_body for advanced parameters
            extra_body = {
                "min_p": min_p,
                'include_stop_str_in_output': True,
                'top_k': top_k,
            }
            
            if schema:
                extra_body["guided_json"] = schema.model_json_schema()
            
            request_params["extra_body"] = extra_body
            
            response = client.chat.completions.create(**request_params)
            
            if stream:
                response_content = ""   
                for chunk in response:
                    response_content += chunk.choices[0].message.content
                    print(chunk.choices[0].message.content, end="", flush=True)
                content = response_content
            else:
                content = response.choices[0].message.content

            if content and content.strip():
                print("--- Service call successful, received a valid response ---")
                return content.strip()
            else:
                print(f"Warning: Attempt {attempt + 1} received an empty response.")

        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
        except Exception as e:
            print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

        if attempt < max_tries - 1:
            sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
            sleep_time = min(sleep_time, 1)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print("Error: All retry attempts have been exhausted. The call has failed.")

    return "vllm server error!!!"

def count_tokens(messages: List[Dict], llm_local_path: str) -> int:
    """Count tokens in messages."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_local_path)
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(full_prompt, return_tensors="pt")
    token_count = len(tokens["input_ids"][0])
    return token_count


def batch_completion(
    messages_list: List, 
    api_base: str,
    model_name: str, 
    llm_generate_cfg: Dict,
    max_tokens: int = 10000, 
    top_p: float = 0.8, 
    temperature: float = 0.7, 
    repetition_penalty: float = 1.05, 
    min_p: float = 0.05, 
    top_k: int = 20
) -> List:
    """
    Process multiple prompts in parallel using ThreadPoolExecutor.
    
    Args:
        messages_list: List of message lists to process
        api_base: Base URL for the API
        model_name: Name of the model to use
        llm_generate_cfg: LLM generation configuration
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        temperature: Temperature for sampling
        repetition_penalty: Repetition penalty
        min_p: Minimum probability threshold
        top_k: Top-k sampling parameter
        
    Returns:
        List of generated responses
    """
    print(f"Processing {len(messages_list)} prompts in parallel...")
    results = [None] * len(messages_list)  # Initialize a list with the same length as prompts
    
    # Define a worker function for threading
    def worker(index, messages):
        result = call_server(
            messages=messages,
            api_base=api_base,
            model_name=model_name,
            llm_generate_cfg=llm_generate_cfg,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_k=top_k,
            stream=False  # Disable streaming for batch processing
        )
        results[index] = result
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map each prompt to the worker function
        futures = {executor.submit(worker, idx, messages): idx for idx, messages in enumerate(messages_list)}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Get task result, exceptions will be caught if thrown
            except Exception as exc:
                print(f'Generated an exception: {exc}')
                idx = futures[future]
                results[idx] = None
    
    return results

def truncate_google_search_content(msg, max_tokens=1024):
    if msg["content"].startswith("<tool_response>\nGoogle Search for queries:"):
        print("truncating google search content")
        n_tokens = num_tokens_from_messages([{"role": msg["role"], "content": msg["content"]}])
        if n_tokens > max_tokens:
            words = msg["content"].split()[:max_tokens]
            msg["content"] = " ".join(words)

def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    pattern = re.escape(end_marker [::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
    matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
    if matches:
        return matches[0][::-1].strip()
    return None


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], file_path: str):
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_json(data: Dict, file_path: str) -> None:
    """Save a dictionary to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def load_json(file_path: str) -> Dict:
    """Load a JSON file into a dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
