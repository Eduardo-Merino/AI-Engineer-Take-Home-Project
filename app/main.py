# app/main.py
"""
FastAPI application exposing a single chat endpoint.

Flow per request:
1. Accept user input + session_id.
2. Run RAG retrieval to gather context.
3. Build system prompt and call Anthropic Claude with tool specs.
4. If the model requests tools, execute them and send a follow‑up call.
5. Return the final assistant answer and record it in session history.

Run:
    uvicorn app.main:app --reload
"""

import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anthropic import Anthropic, APIStatusError

from app.rag.retriever import build_context
from app.rag.tools import get_tool_specs, execute_tool
from app.rag.prompt import build_system_prompt
from app.sessions import SessionManager
from app.logging_config import configure_logging
import logging

# ---------------------------
# Configuration
# ---------------------------
configure_logging()
logger = logging.getLogger(__name__)
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 500

# Instantiate global singletons
session_manager = SessionManager()
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

app = FastAPI(title="Simple RAG + Tool Agent", version="1.0.0")


class ChatRequest(BaseModel):
    """
    Input schema for the /chat endpoint.

    Attributes
    ----------
    session_id : str
        Arbitrary identifier grouping consecutive interactions.
    message : str
        Natural language user query.
    """
    session_id: str = Field(..., description="Session identifier.")
    message: str = Field(..., description="User input text.")


class ChatResponse(BaseModel):
    """
    Output schema for the /chat endpoint.

    Attributes
    ----------
    answer : str
        Final assistant answer after optional tool usage.
    tool_calls : list[dict]
        List of tool invocations (name + arguments) if any were executed.
    """
    answer: str
    tool_calls: List[Dict[str, Any]] = []


def _format_history_for_llm(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert internal history (role/content) to Anthropic message objects.

    Parameters
    ----------
    history : list[dict]
        Stored session messages.

    Returns
    -------
    list[dict]
        Messages suitable for `client.messages.create`.

    Functionality
    -------------
    1. Anthropic expects `messages=[{"role":"user"|"assistant","content": "..."}]`.
    2. We pass text content directly; tool interactions are not persisted.
    """
    return [{"role": m["role"], "content": m["content"]} for m in history]


def _extract_text_from_blocks(blocks: List[Any]) -> str:
    """
    Concatenate all text content blocks from Claude response.

    Parameters
    ----------
    blocks : list
        Response.content list from Anthropic.

    Returns
    -------
    str
        Concatenated text.

    Functionality
    -------------
    1. Iterates over block objects.
    2. Collects `.text` for those with type == "text".
    3. Joins them with newlines.
    """
    texts = []
    for b in blocks:
        if getattr(b, "type", None) == "text":
            texts.append(b.text)
    return "\n".join(texts).strip()


def _run_llm_with_tools(system_prompt: str,
                        messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute an LLM call with tool support (one round, plus optional tool follow‑up).

    Parameters
    ----------
    system_prompt : str
        System prompt including retrieved context.
    messages : list[dict]
        Prior chat history + current user message.

    Returns
    -------
    dict
        Keys:
        - "answer": final assistant natural language answer.
        - "tool_calls": list of executed tool metadata.

    Functionality
    -------------
    1. Calls Claude with tool specifications.
    2. If Claude emits tool_use blocks:
       a. Execute each tool locally.
       b. Append the tool outputs as `tool_result` blocks.
       c. Issue a second Claude call to get a final answer.
    3. If no tools requested, use the direct text response.
    """
    tool_specs = get_tool_specs()

    # First request
    first_response = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        tools=tool_specs,
    )

    tool_calls: List[Dict[str, Any]] = []
    tool_results_blocks = []

    # Inspect content for tool_use blocks
    for block in first_response.content:
        if block.type == "tool_use":
            # Execute tool
            try:
                result = execute_tool(block.name, block.input)
            except Exception as exc:  # tool failure fallback
                result = {"error": str(exc)}
            tool_calls.append({"name": block.name, "arguments": block.input, "result": result})

            # Prepare a tool_result block referencing the tool_use block.id
            tool_results_blocks.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result)
            })

    # If there were tool calls, send a follow‑up request
    if tool_results_blocks:
        followup_messages = messages + [
            {
                "role": "assistant",
                "content": first_response.content,  # original tool_use response
            },
            {
                "role": "user",
                "content": tool_results_blocks,     # feed tool outputs back
            },
        ]
        second_response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=followup_messages,
        )
        answer_text = _extract_text_from_blocks(second_response.content)
    else:
        # No tools used; just extract text from the first response
        answer_text = _extract_text_from_blocks(first_response.content)

    return {"answer": answer_text, "tool_calls": tool_calls}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint implementing RAG + tool usage.

    Parameters
    ----------
    req : ChatRequest
        Incoming request body containing session_id and user message.

    Returns
    -------
    ChatResponse
        Final assistant answer plus any tool call metadata.

    Functionality
    -------------
    1. Append the user message to session history.
    2. Build retrieval context using the latest user query.
    3. Build system prompt with context block.
    4. Format full history for the LLM.
    5. Run the LLM with tool loop.
    6. Persist the assistant's answer into the session.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    logger.info("New message session=%s text=%r", req.session_id, req.message)
    # Add user message to history
    session_manager.append_user(req.session_id, req.message)

    # Build retrieval context from the *current* user query
    context_block = build_context(req.message, k=3)
    system_prompt = build_system_prompt(context_block)

    # Prepare messages for the model
    history = session_manager.get_history(req.session_id)
    messages = _format_history_for_llm(history)

    try:
        result = _run_llm_with_tools(system_prompt, messages)
    except APIStatusError as api_err:
        raise HTTPException(status_code=500, detail=f"Anthropic API error: {api_err}") from api_err

    # Persist assistant answer
    session_manager.append_assistant(req.session_id, result["answer"])

    return ChatResponse(answer=result["answer"], tool_calls=result["tool_calls"])
