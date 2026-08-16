from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
from src.api.v1.schemas.query_schema import (
    QueryRequest,
    QueryResponse,
)
from src.api.v1.agents.agents import banking_agent
from src.api.v1.services.query_service import (
    query_documents,
    query_documents_stream,
)

router = APIRouter(
    prefix="/api/v1/query",
    tags=["Query"],
)


# NORMAL QUERY
@router.post(
    "/",
    response_model=QueryResponse,
)
async def query(request: QueryRequest):
    return query_documents(
        request.question,
        account_id=request.account_id,
        thread_id=request.thread_id,
    )


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
):
    def event_generator():
        try:
            for event in query_documents_stream(
                question=request.question,
                account_id=request.account_id,
                thread_id=request.thread_id,
            ):
                yield (f"data: {json.dumps(event, default=str)}\n\n")
        except Exception as error:
            print("STREAM ROUTE ERROR:", repr(error))
            error_event = {"type": "error", "message": str(error)}
            yield (f"data: {json.dumps(error_event)}\n\n")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
