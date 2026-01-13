# MemoBrain Examples

This directory contains examples demonstrating how to use MemoBrain for managing long-horizon reasoning in tool-augmented agents.

## 📋 Overview

The examples showcase a **ReAct agent** integrated with **MemoBrain** for automatic memory management during complex reasoning tasks. The agent can:

- Perform web searches and visit pages
- Execute Python code
- Manage context automatically with MemoBrain
- Handle long-horizon reasoning tasks efficiently

## 🚀 Quick Start

### 1. Model Deployment

Deploy three models using vLLM in separate terminals:

```bash
# Terminal 1: Reasoning Model (Main Agent)
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --port 8000

# Terminal 2: Auxiliary Model (Web Page Summarization)
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001

# Terminal 3: Memory Model (Memory Management)
vllm serve TommyChien/MemoBrain-14B --port 8002
```

**Model Roles:**
- **Reasoning Model**: Main agent for complex reasoning and decision-making
- **Auxiliary Model**: Extracts relevant evidence from web pages
- **Memory Model**: Manages reasoning trajectory (flush, fold, recall operations)

### 2. Configure API Keys

Set up your API keys for external services:

```bash
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CX="your_google_cx"
export JINA_API_KEY="your_jina_api_key"
```

### 3. Run Evaluations

```bash
# Run with MemoBrain (default)
python run_task.py --eval_task gaia

# Run without MemoBrain (vanilla ReAct)
python run_task.py --eval_task gaia --no_memory

# Run other tasks
python run_task.py --eval_task ww    # WebWalker
python run_task.py --eval_task bcp   # BrowseComp
```

## 📁 File Structure

```
examples/
├── README.md                    # This file
├── config.py                    # Configuration management
├── state.py                     # Agent state definition
├── prompts.py                   # System prompts
├── tools.py                     # Tool implementations (Search, Visit, etc.)
├── react_with_memory.py         # ReAct agent with optional memory
├── run_task.py                  # Task runner for evaluations
├── utils.py                     # Utility functions
└── example_usage.py             # Simple MemoBrain usage example
```

## 🎯 Core Components

### ReAct Agent (`react_with_memory.py`)

The ReAct agent implements a reasoning loop with tool execution. Key features:

- **Planning**: LLM generates reasoning and tool calls
- **Tool Execution**: Executes search, visit, and code execution tools
- **Memory Management** (optional): Automatic context optimization with MemoBrain
- **Token Budget**: Monitors context size and triggers memory optimization

**Usage:**

```python
import asyncio
from config import Configuration
from react_with_memory import run_react_agent

async def main():
    config = Configuration.from_runnable_config()
    
    # With memory (default)
    result = await run_react_agent(
        question="Your complex question here",
        config=config,
        use_memory=True
    )
    
    # Without memory
    result = await run_react_agent(
        question="Your question",
        config=config,
        use_memory=False
    )

asyncio.run(main())
```

### Configuration (`config.py`)

Manages all configuration parameters:

```python
from config import Configuration

config = Configuration.from_runnable_config()

# Key parameters:
# - reasoning_model: Main reasoning model
# - auxiliary_model: Web page summarization model
# - memory_model: Memory management model
# - max_memory_size: Token budget for context (default: 32K)
# - max_llm_call_per_run: Maximum LLM calls (default: 200)
# - use_memory: Whether to use MemoBrain (default: True)
```

### Tools (`tools.py`)

Available tools for the agent:

1. **Search**: Google Custom Search API
   - Multi-query search
   - Result ranking and filtering
   
2. **Visit**: Jina API for webpage fetching
   - Markdown conversion
   - Evidence extraction
   
3. **PythonInterpreter**: Code execution
   - Safe sandbox execution
   - Output capture

## 💡 Usage Examples

### Example 1: Run with MemoBrain

```bash
python run_task.py \
  --eval_task gaia \
  --reasoning_model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --reasoning_model_base_url http://localhost:8000/v1 \
  --auxiliary_model Qwen/Qwen2.5-14B-Instruct \
  --auxiliary_model_base_url http://localhost:8001/v1 \
  --memory_model TommyChien/MemoBrain-14B \
  --memory_model_base_url http://localhost:8002/v1 \
  --max_memory_size 32768
```

### Example 2: Run without MemoBrain

```bash
python run_task.py \
  --eval_task gaia \
  --no_memory \
  --reasoning_model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --reasoning_model_base_url http://localhost:8000/v1
```

### Example 3: Custom Configuration

```python
from config import Configuration

# Create custom configuration
config = Configuration(
    reasoning_model="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    reasoning_model_base_url="http://localhost:8000/v1",
    reasoning_model_api_key="empty",
    auxiliary_model="Qwen/Qwen2.5-14B-Instruct",
    auxiliary_model_base_url="http://localhost:8001/v1",
    auxiliary_model_api_key="empty",
    memory_model="TommyChien/MemoBrain-14B",
    memory_model_base_url="http://localhost:8002/v1",
    memory_model_api_key="empty",
    max_memory_size=32*1024,  # 32K tokens
    max_llm_call_per_run=200,
    use_memory=True
)
```

### Example 4: Programmatic Evaluation

```python
import asyncio
from config import Configuration
from react_with_memory import run_react_agent

async def evaluate_question():
    config = Configuration.from_runnable_config()
    
    question = "What is the population of the capital of France?"
    
    result = await run_react_agent(
        question=question,
        answer="",  # Ground truth (optional)
        config=config,
        use_memory=True
    )
    
    print(f"Question: {result['question']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Token Count: {result['token_count']}")
    print(f"Total Time: {result['total_process_time']:.2f}s")
    print(f"Memorize Time: {result['total_memorize_time']:.2f}s")
    print(f"Recall Time: {result['total_recall_time']:.2f}s")

asyncio.run(evaluate_question())
```

## 🔧 Command-line Arguments

### Common Arguments

```bash
--eval_task TASK              # Task to evaluate (gaia, ww, bcp)
--version VERSION             # Version identifier for results
--use_memory / --no_memory   # Enable/disable MemoBrain (default: enabled)
```

### Model Configuration

```bash
--reasoning_model MODEL       # Main reasoning model
--reasoning_model_base_url URL
--reasoning_model_api_key KEY

--auxiliary_model MODEL       # Auxiliary model for summarization
--auxiliary_model_base_url URL
--auxiliary_model_api_key KEY

--memory_model MODEL          # Memory management model
--memory_model_base_url URL
--memory_model_api_key KEY
```

### Memory Configuration

```bash
--max_memory_size SIZE        # Max tokens in context (default: 32768)
--max_llm_call_per_run N      # Max LLM calls (default: 200)
```

### API Keys

```bash
--google_api_key KEY          # Google Custom Search API key
--google_cx CX                # Google Custom Search engine ID
--jina_api_key KEY            # Jina API key
```

## 📊 Evaluation Tasks

### GAIA (General AI Assistant)

```bash
python run_task.py --eval_task gaia
```

Evaluates on GAIA benchmark questions requiring multi-step reasoning and tool use.

### WebWalker

```bash
python run_task.py --eval_task ww
```

Tests web navigation and information extraction capabilities.

### BrowseComp

```bash
python run_task.py --eval_task bcp
```

Evaluates browsing and comprehension tasks.

## 🎓 How It Works

### 1. Agent Initialization

```python
# Initialize MemoBrain (if enabled)
if use_memory:
    memory = MemoBrain(
        api_key=config.memory_model_api_key,
        base_url=config.memory_model_base_url,
        model_name=config.memory_model
    )
    memory.init_memory(question)
```

### 2. Reasoning Loop

```
┌─────────────────────────────────────────┐
│          Planning Node                  │
│  - Generate reasoning                   │
│  - Decide on tool calls                 │
│  - Check token budget                   │
│  - Trigger recall if needed             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Tool Call Node                 │
│  - Execute tool (search/visit/code)     │
│  - Get tool response                    │
│  - Memorize episode (if memory enabled) │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       Check Limits Node                 │
│  - Check token count                    │
│  - Force answer if limit exceeded       │
└─────────────────────────────────────────┘
                    ↓
              Continue or End
```

### 3. Memory Management

When context exceeds `max_memory_size`:

```python
if memory and current_context_size > max_memory_size:
    # Trigger MemoBrain recall
    optimized_messages = await memory.recall()
    # Use optimized messages for next iteration
```

### 4. Episode Recording

After each tool call:

```python
if memory:
    # Record reasoning + tool response as an episode
    await memory.memorize([
        assistant_message,  # Agent's reasoning
        tool_response       # Tool execution result
    ])
```

## 📈 Performance Metrics

The evaluation tracks:

- **Token Count**: Total tokens used in conversation
- **Process Time**: Total execution time
- **Memorize Time**: Time spent on memory recording
- **Recall Time**: Time spent on memory optimization
- **Prediction**: Agent's final answer
- **Termination**: How the agent finished (answer/timeout/limit)

## 🔍 Debugging

### Enable Verbose Output

```python
# In planning_node
print(f'Round {state["round"] + 1}: {content}')
print(f"token count: {token_count}")
```

### Inspect Memory Graph

```python
if 'memory' in result:
    memory_graph = result['memory']
    print(f"Nodes: {len(memory_graph['nodes'])}")
    print(f"Edges: {len(memory_graph['edges'])}")
```

### Save Intermediate States

```python
# Save messages to file
import json
with open(f"debug_round_{round}.jsonl", "w") as f:
    for msg in messages:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
```

## ⚙️ Advanced Usage

### Custom Memory Strategy

```python
# Adjust memory budget based on task complexity
if task_complexity == "high":
    config.max_memory_size = 64 * 1024  # 64K tokens
else:
    config.max_memory_size = 16 * 1024  # 16K tokens
```

### Multiple Memory Models

```python
# Use different memory models for different tasks
if eval_task == "gaia":
    config.memory_model = "TommyChien/MemoBrain-14B"
else:
    config.memory_model = "TommyChien/MemoBrain-8B"
```

### Custom Tools

Add new tools in `tools.py`:

```python
class MyCustomTool:
    def __init__(self):
        self.name = "my_tool"
        self.description = "Description of my tool"
    
    def call(self, params: dict, **kwargs) -> str:
        # Tool implementation
        return result

# Add to TOOL_CLASS
TOOL_CLASS = [
    Search(),
    Visit(),
    PythonInterpreter(),
    MyCustomTool(),  # Your tool
]
```

## 📝 Notes

- **Memory Overhead**: MemoBrain adds ~2-5% overhead for memory operations
- **Token Savings**: Can reduce context by 30-50% on long tasks
- **Model Requirements**: Memory model needs ~14GB VRAM (for MemoBrain-14B)
- **API Costs**: Using MemoBrain reduces API costs by optimizing context

## 🤝 Support

For issues or questions:
- Check the main [README](../README.md)
- Review configuration in `config.py`
- Inspect tool implementations in `tools.py`
- Examine the agent logic in `react_with_memory.py`

## 📚 Additional Resources

- [MemoBrain Paper]()
- [Model Cards on Hugging Face](https://huggingface.co/TommyChien)
- [Configuration Guide](config.py)
- [Tool Documentation](tools.py)
