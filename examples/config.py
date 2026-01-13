import os
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict

from langchain_core.runnables import RunnableConfig
import argparse

class ConfigParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--reasoning_model", type=str, default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
        self.parser.add_argument("--reasoning_model_api_key", type=str, default="empty")
        self.parser.add_argument("--reasoning_model_base_url", type=str, default="http://localhost:8000/v1")
        self.parser.add_argument("--auxiliary_model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
        self.parser.add_argument("--auxiliary_model_base_url", type=str, default="http://localhost:8001/v1")
        self.parser.add_argument("--auxiliary_model_api_key", type=str, default="empty")
        self.parser.add_argument("--memory_model", type=str, default="TommyChien/MemoBrain-14B")
        self.parser.add_argument("--memory_model_base_url", type=str, default="http://localhost:8002/v1")
        self.parser.add_argument("--memory_model_api_key", type=str, default="empty")
        self.parser.add_argument("--google_api_key", type=str, default="")
        self.parser.add_argument("--google_cx", type=str, default="")
        self.parser.add_argument("--jina_api_key", type=str, default="")
        self.parser.add_argument("--max_memory_size", type=int, default=32*1024)
        self.parser.add_argument("--max_llm_call_per_run", type=int, default=200)
        self.parser.add_argument("--eval_task", type=str, default="GAIA")
        self.parser.add_argument("--version", type=str, default="v1")
        self.parser.add_argument("--use_memory", action='store_true', default=True)
        self.parser.add_argument("--no_memory", dest='use_memory', action='store_false')
        self.args = self.parser.parse_args()



class Configuration(BaseModel):
    """The configuration for the agent."""

    reasoning_model: str = Field(
        default="",
        metadata={
            "description": "The name of the language model to use for the agent's reasoning."
        },
    )
    reasoning_model_api_key: str = Field(
        default="",
        metadata={
            "description": "The API key of the language model to use for the agent's reasoning."
        },
    )
    reasoning_model_base_url: str = Field(
        default="",
        metadata={
            "description": "The base URL of the language model to use for the agent's reasoning."
        },
    )
    auxiliary_model_api_key: str = Field(
        default="",
        metadata={
            "description": "The API key of the language model to use for the agent's auxiliary."
        },
    )
    auxiliary_model: str = Field(
        default="",
        metadata={
            "description": "The name of the language model to use for the agent's auxiliary."
        },
    )
    auxiliary_model_base_url: str = Field(
        default="",
        metadata={
            "description": "The base URL of the language model to use for the agent's auxiliary."
        },
    )
    memory_model: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The name of the language model to use for memory operations."
        },
    )
    memory_model_base_url: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The base URL of the language model to use for memory operations."
        },
    )
    memory_model_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The API key of the language model to use for memory operations."
        },
    )
    google_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Google Custom Search API key."
        },
    )
    google_cx: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Google Custom Search engine ID."
        },
    )
    jina_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Jina API key for web page fetching."
        },
    )
    max_llm_call_per_run: int = Field(
        default=100,
        metadata={"description": "The maximum number of LLM calls per run."},
    )
    max_retries: int = Field(
        default=3,
        metadata={"description": "The maximum number of retries for API calls."},
    )
    eval_task: str = Field(
        default="GAIA",
        metadata={"description": "The task to evaluate."},
    )
    version: str = Field(
        default="v1",
        metadata={"description": "The version of the evaluation."},
    )
    llm_generate_cfg: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 1.1
        },
        metadata={"description": "The LLM generation configuration."},
    )
    max_tokens: int = Field(
        default=105 * 1024,
        metadata={"description": "The maximum number of tokens."},
    )
    max_time_seconds: int = Field(
        default=150 * 60,
        metadata={"description": "The maximum time in seconds (150 minutes)."},
    )
    max_memory_size: int = Field(
        default=32*1024,
        metadata={"description": "The maximum number of tokens in the memory."},
    )
    use_memory: bool = Field(
        default=True,
        metadata={"description": "Whether to use MemoBrain for memory management."},
    )
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        parser = ConfigParser()

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        for k, v in parser.args.__dict__.items():
            values[k] = v

        return cls(**values)
