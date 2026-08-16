import json
from src.api.v1.agents.agents import banking_agent


def build_initial_state(
    question: str,
    account_id: str | None = None,
):
    return {
        "question": question,
        "search_query": question,
        "query_type": "",
        "account_id": account_id,
        "retrieved_chunks": [],
        "fts_chunks": [],
        "hybrid_chunks": [],
        "reranked_chunks": [],
        "rewritten_queries": [],
        "sql_query": "",
        "validated_sql": "",
        "sql_result": [],
        "answer": "",
        "citations": [],
        "response_sources": [],
        "confidence_score": 0,
        "retry_count": 0,
        "max_retries": 2,
        "trace_id": "",
    }


def query_documents(
    question: str,
    account_id: str | None = None,
    thread_id: str | None = None,
):
    if not thread_id:
        raise ValueError("thread_id is required.")
    initial_state = build_initial_state(
        question=question,
        account_id=account_id,
    )
    config = get_thread_config(thread_id)
    response = banking_agent.invoke(
        initial_state,
        config=config,
    )
    return {
        "answer": response.get("answer", ""),
        "query_type": response.get("query_type", ""),
        "citations": response.get("citations", []),
        "images": response.get("response_sources", []),
        "confidence_score": response.get("confidence_score", 0),
    }


def get_node_message(node_name: str):
    messages = {
        "classifier": "Understanding your query...",
        "conversation": "Continuing our conversation...",
        "out_of_scope": "Checking the query scope...",
        "vector_search": "Searching the knowledge base...",
        "fts_search": "Running keyword search...",
        "hybrid_search": "Combining search results...",
        "rerank": "Reranking the most relevant information...",
        "reranker": "Reranking the most relevant information...",
        "retry": "Refining the search query...",
        "retry_search": "Refining the search query...",
        "sql": "Checking banking database records...",
        "sql_generator": "Preparing database query...",
        "sql_validator": "Validating database query...",
        "sql_executor": "Executing database query...",
        "merge_context": "Combining banking records and knowledge...",
        "response_generator": "Generating the final answer...",
    }
    return messages.get(node_name, f"Processing: {node_name}")


def query_documents_stream(
    question: str,
    account_id: str | None = None,
    thread_id: str | None = None,
):
    if not thread_id:
        raise ValueError("thread_id is required.")
    initial_state = build_initial_state(
        question=question,
        account_id=account_id,
    )
    config = get_thread_config(thread_id)
    yield {
        "type": "status",
        "message": "Understanding your query...",
    }
    final_state = initial_state.copy()
    try:
        for update in banking_agent.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            # print("GRAPH UPDATE:", update)
            if not update:
                continue
            for node_name, node_state in update.items():
                if isinstance(node_state, dict):
                    final_state.update(node_state)
                yield {
                    "type": "status",
                    "node": node_name,
                    "message": get_node_message(node_name),
                }
        # print("FINAL STATE:", final_state)
        answer = final_state.get("answer", "")
        # print("FINAL ANSWER:", repr(answer))
        yield {
            "type": "complete",
            "answer": answer,
            "query_type": final_state.get("query_type", ""),
            "citations": final_state.get("citations", []),
            "images": final_state.get("response_sources", []),
            "confidence_score": final_state.get("confidence_score", 0),
        }
    except Exception as e:
        print(
            "STREAMING GRAPH ERROR:",
            repr(e),
        )
        yield {
            "type": "error",
            "message": str(e),
        }


def get_thread_config(thread_id: str):
    """
    LangGraph checkpoint configuration.
    The thread_id identifies one conversation.
    """
    if not thread_id:
        raise ValueError("thread_id is required.")
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }
