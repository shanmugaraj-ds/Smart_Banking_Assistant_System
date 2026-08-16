from typing import TypedDict, List, Dict, Any
from typing import Annotated
from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    question: str  # User Input
    search_query: str
    account_id: str | None
    query_type: str
    chat_history: Annotated[list, add_messages]
    retrieved_chunks: List[Dict[str, Any]]  # RAG Pipeline
    fts_chunks: List[Dict[str, Any]]  # RAG Pipeline
    hybrid_chunks: List[Dict[str, Any]]  # RAG Pipeline
    reranked_chunks: List[Dict[str, Any]]  # RAG Pipeline
    rewritten_queries: List[str]  # RAG Pipeline
    sql_query: str  # SQL Pipeline
    sql_result: List[Dict[str, Any]]
    validated_sql: str
    answer: str  # Final Response
    citations: List[str]  # Final Response
    response_sources: List[str]
    confidence_score: float  # Final Response
    retry_count: int  # Retry
    trace_id: str  # LangSmith
    max_retries: int
    final_context: dict[str, Any]
