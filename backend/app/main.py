from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from .llm import get_llm_provider, LLMGenerationError


logger = logging.getLogger("chatbot")
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="No-Memory Chatbot API", version="1.0.0")

# CORS for local dev frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = get_llm_provider()


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        reply_text = llm.generate_sync(
            user_message=req.message,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return ChatResponse(reply=reply_text)
    except LLMGenerationError as e:
        logger.exception("LLM generation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream")
def stream(message: str = Query(..., min_length=1),
           max_new_tokens: int = Query(256, ge=1, le=2048),
           temperature: float = Query(0.7, ge=0.0, le=2.0),
           top_p: float = Query(0.9, ge=0.0, le=1.0)):
    try:
        def event_generator():
            try:
                for token in llm.generate_stream(
                    user_message=message,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    yield {"event": "message", "data": token}
                yield {"event": "end", "data": ""}
            except LLMGenerationError as e:
                yield {"event": "error", "data": str(e)}
        return EventSourceResponse(event_generator())
    except Exception as e:
        logger.exception("Streaming setup failure")
        raise HTTPException(status_code=500, detail=str(e))