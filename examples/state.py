from typing import Dict, List, TypedDict, Annotated, Optional
from memobrain import MemoBrain

def add_messages(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Reducer function for messages in langgraph."""
    return left + right

def replace_or_add_messages(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Reducer function for compressed_messages: replace if right starts with __REPLACE__ marker, else add."""
    if right and right[0] == "__REPLACE__":
        return right[1:]
    return left + right

class OverallState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    compressed_messages: Annotated[List[Dict], replace_or_add_messages]
    web_pages: Dict[str, str]
    response: str
    status: List[str]
    question: str
    round: int
    num_llm_calls_available: int
    start_time: float
    prediction: str
    termination: str
    answer: str
    memory: Optional[MemoBrain]
    token_count: int
    current_working_context_size: int
    total_memorize_time: float
    total_recall_time: float
    total_process_time: float
