import json5
import asyncio
from typing import Dict, Optional
import requests
import tiktoken
from typing import List, Union
from config import Configuration
from prompts import EXTRACTOR_PROMPT
from openai import OpenAI
import json
import time
import re
from requests.exceptions import Timeout


class Search:
    """Search tool using Google Custom Search API."""
    
    def __init__(self, api_key: str = None, cx: str = None):
        self.name = "search"
        self.description = "Performs online web searches using Google Custom Search API: supply 'query' (string or array); returns top search results."
        self.api_url = "https://www.googleapis.com/customsearch/v1"
        self.api_key = api_key
        self.cx = cx
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def truncate_to_tokens(self, text: str, max_tokens: int = 128) -> str:
        """Truncate text to specified number of tokens."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def _google_search(self, query: str, timeout: int = 20) -> dict:
        """
        Execute a single Google search query.
        
        Args:
            query: Search query string
            timeout: Request timeout in seconds
            
        Returns:
            Search results dictionary
        """
        params = {
            'q': query,
            'key': self.api_key,
            'cx': self.cx
        }
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(self.api_url, params=params, timeout=timeout)
                response.raise_for_status()
                search_results = response.json()
                return search_results
            except Timeout:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Google Search request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                    return {}
                print(f"Google Search Timeout occurred, retrying ({retry_count}/{max_retries})...")
            except requests.exceptions.RequestException as e:
                print(f"Google Search Request Error occurred: {e}")
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Google Search Request Error occurred: {e} after {max_retries} retries")
                    return {}
                print(f"Google Search Request Error occurred, retrying ({retry_count}/{max_retries})...")
            time.sleep(1)
        
        return {}
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Call the Google Custom Search API with queries.
        
        Args:
            params: Dictionary containing 'query' field with search query (string or list)
            
        Returns:
            Formatted search results as string
        """
        try:
            if isinstance(params, dict) and "query" in params:
                queries = params["query"]
            else:
                return "[Search] Invalid request format: Input must contain 'query' field"
            
            # Ensure queries is a list
            if isinstance(queries, str):
                queries = [queries]


            # Fetch all queries
            formatted_results = []
            url_2_text = {}
            seen_urls = set()
            all_queries = []
            all_results = []
            
            for query in queries:
                all_queries.append(query)
                data = self._google_search(query)
                
                if not data or "items" not in data:
                    continue
                
                results = data.get("items", [])
                
                for result in results:
                    url = result.get("link", "")
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        url_2_text[url] = snippet
                        truncated_text = self.truncate_to_tokens(snippet, 128)
                        all_results.append((url, title, truncated_text))
            
            # Compose header
            queries_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(all_queries))
            query_output = f"Google Search for queries:\n{queries_str}\n\nResults:\n\n## Web Results\n"
            for i, (url, title, truncated_text) in enumerate(all_results, 1):
                query_output += f"{i}. {title}\n   URL: {url}\n   {truncated_text}\n\n"
            formatted_results.append(query_output)
            
            return {"results": "\n=======\n".join(formatted_results), "meta": url_2_text}
            
        except Exception as e:
            return f"[Search] Error: {str(e)}"


class Visit:
    """Visit tool using Jina API to fetch webpage content and extract evidence."""
    
    def __init__(self, jina_api_key: str = None):
        self.name = "visit"
        self.description = "Visits webpages and extracts relevant evidence based on a goal: supply 'urls' (string or array) and 'goal' (string)."
        self.jina_api_key = jina_api_key
    
    def call_server(self, msgs, api_base, model_name, api_key, max_retries=2):
        """Call LLM server to extract evidence."""
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception as e:
                print(f"[Visit] Error: {str(e)}")
                if attempt == (max_retries - 1):
                    return f"[Visit] Error: {str(e)} (Max retries reached)"
                continue
    
    def _fetch_jina_content(self, url: str, keep_links: bool = False) -> str:
        """
        Fetch webpage content using Jina API.
        
        Args:
            url: URL to fetch
            keep_links: Whether to keep links in the content
            
        Returns:
            Webpage content in markdown format
        """
        jina_headers = {
            'Authorization': f'Bearer {self.jina_api_key}',
            'X-Return-Format': 'markdown',
        }
        
        try:
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers, timeout=30)
            response.raise_for_status()
            text = response.text
            
            if not keep_links:
                pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                text = re.sub(pattern, "", text)
            
            text = text.replace('---', '-').replace('===', '=').replace('   ', ' ').replace('   ', ' ')
            return text
        except Exception as e:
            return f"[Visit] Error fetching content from {url}: {str(e)}"
    
    def call(self, params: Union[str, dict], config: Configuration = None, **kwargs) -> str:
        """
        Call the Jina API to fetch webpage content and extract evidence.
        
        Args:
            params: Dictionary containing:
                - 'urls' or 'url': URL(s) to visit (string or list)
                - 'goal': (optional) Goal for evidence extraction
            config: Configuration object for LLM settings
            
        Returns:
            Extracted evidence as string, or error message
        """
        try:
            # Parse URLs
            if isinstance(params, dict) and "urls" in params:
                urls = params["urls"]
            elif isinstance(params, dict) and "url" in params:
                urls = params["url"]
            else:
                return "[Visit] Invalid request format: Input must contain 'urls' or 'url' field"
            
            # Parse goal (optional)
            goal = params.get("goal", "") if isinstance(params, dict) else ""
            
            # Ensure urls is a list
            if isinstance(urls, str):
                urls = [urls]
            
            # Fetch content from all URLs using Jina
            results_dict = {}
            for url in urls:
                content = self._fetch_jina_content(url)
                results_dict[url] = content
            
            # If no goal provided, just return the raw content
            if not goal:
                # Format the content for display
                formatted_output = []
                for url, content in results_dict.items():
                    formatted_output.append(f"URL: {url}\n\nContent:\n{content}\n")
                return "\n" + "="*80 + "\n\n".join(formatted_output)
            
            # Extract evidence using LLM if goal is provided
            if not config:
                return "[Visit] Configuration required for evidence extraction"
            
            model_name = params.get("model_name", config.auxiliary_model)
            api_url = params.get("api_url", config.auxiliary_model_base_url)
            api_key = params.get("api_key", config.auxiliary_model_api_key)
            
            all_evidence = []
            for url, content in results_dict.items():
                messages = [{
                    "role": "user", 
                    "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
                }]
                
                # Call LLM to extract evidence
                try:
                    evidence = self.call_server(messages, api_base=api_url, model_name=model_name, api_key=api_key, max_retries=2)
                except Exception as e:
                    evidence = f"[Visit] Error extracting evidence: {str(e)}"
                
                all_evidence.append(f"URL: {url}\nEvidence: {evidence.strip()}")
            
            # Combine all evidence
            combined_evidence = "\n\n---\n\n".join(all_evidence)
            return combined_evidence
            
        except Exception as e:
            return f"[Visit] Error: {str(e)}"

class PythonInterpreter:
    """
    A simple Python interpreter tool that executes Python code and returns the output or error.
    """
    def __init__(self):
        self.name = "PythonInterpreter"
        self.description = "Executes Python code and returns the output or errors."

    def call(self, code_or_args):
        """
        Executes the provided Python code string in a restricted namespace.

        Args:
            code_or_args: Python code (str), or a dict with a 'code' key

        Returns:
            Output (str) or error message
        """
        import sys
        import io
        import contextlib

        # Accept both raw code string or a dict with 'code' key (for compatibility)
        code = None
        if isinstance(code_or_args, dict) and "code" in code_or_args:
            code = code_or_args["code"]
        elif isinstance(code_or_args, str):
            code = code_or_args
        else:
            return "[Python Interpreter Error]: No code provided."

        output = ""
        # Minimal globals: no __import__, no modules, no access to files or system
        # Create a fake print to capture output
        stdout_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(
                    code,
                    {'__builtins__': __builtins__},
                    {}
                )
        except Exception as e:
            return f"[Python Interpreter Error]: {e}"
        output = stdout_buffer.getvalue().strip()
        if not output:
            output = "[Python Interpreter]: No output."
        return output


def init_tools(google_api_key: str = None, google_cx: str = None, jina_api_key: str = None):
    """
    Initialize tools with API keys.
    
    Args:
        google_api_key: Google Custom Search API key
        google_cx: Google Custom Search engine ID
        jina_api_key: Jina API key
        
    Returns:
        List of initialized tool instances
    """
    return [
        Search(api_key=google_api_key, cx=google_cx),
        Visit(jina_api_key=jina_api_key),
        PythonInterpreter(),
    ]

# Default tool class list (without API keys)
TOOL_CLASS = [
    Search(),
    Visit(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

def custom_call_tool(tool_name: str, tool_args: dict, config: Configuration, web_pages: Dict[str, str], **kwargs) -> str:
    """
    Call a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool
        config: Configuration object
        web_pages: Dictionary of web pages
        **kwargs: Additional keyword arguments
        
    Returns:
        Result from the tool call as a string
    """
    if tool_name not in TOOL_MAP:
        return f"Error: Tool {tool_name} not found"
    
    tool_args["params"] = tool_args

    if "python" in tool_name.lower():
        result = TOOL_MAP['PythonInterpreter'].call(tool_args)
    elif tool_name == "visit":
        # Visit fetches content using Jina and optionally extracts evidence
        result = TOOL_MAP[tool_name].call(tool_args, config=config, **kwargs)
    else:
        raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
        result = raw_result
    return result

def parse_tool_call(content: str) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """
    Parse tool call from content string.
    
    Args:
        content: Content string that may contain tool call
        
    Returns:
        Tuple of (tool_name, tool_args, code) if found, else (None, None, None)
    """
    if '<tool_call>' not in content or '</tool_call>' not in content:
        return None, None, None
    # print("content: ", content)
    tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
    # print("tool_call_str: ", tool_call_str)
    # Check if it's a Python call
    if "python" in tool_call_str.lower() and '<code>' in content and '</code>' in content:
        try:
            code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
            return "PythonInterpreter", {}, code_raw
        except:
            return None, None, None
    
    # Parse JSON tool call
    try:
        tool_call = json5.loads(tool_call_str)
        # print("tool_call: ", tool_call)
        tool_name = tool_call.get('name', '')
        tool_args = tool_call.get('arguments', {})
        return tool_name, tool_args, None
    except:
        return None, None, None

def execute_tool_call(content: str, config: Configuration, web_pages: Dict[str, str]) -> str:
    """
    Execute a tool call from content string.
    
    Args:
        content: Content string containing tool call
        config: Configuration object
        web_pages: Dictionary of web pages
        
    Returns:
        Result from tool execution
    """
    tool_name, tool_args, code = parse_tool_call(content)
    if tool_name is None:
        return 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
    
    if tool_name == "PythonInterpreter" and code:
        try:
            result = TOOL_MAP['PythonInterpreter'].call(code)
        except:
            result = "[Python Interpreter Error]: Formatting error."
    else:
        result = custom_call_tool(tool_name, tool_args, config=config, web_pages=web_pages)
    
    return result
