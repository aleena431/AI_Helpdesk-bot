"""FastAPI WebSocket chat endpoint with per-connection memory and safe streaming."""

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq

from bot import SYSTEM_PROMPT, find_local_faq

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
client = AsyncGroq(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="Alish IT Helpdesk Assistant")

WEBSOCKET_SYSTEM_PROMPT = SYSTEM_PROMPT + """
In this WebSocket demo, create_ticket makes only an in-memory mock ticket. Use it
only when the employee explicitly asks to create or file a ticket. After a
successful tool result, give the mock ID and clearly state that no real IT
system was contacted and no real ticket was submitted.
"""

MOCK_TICKETS: dict[str, dict[str, Any]] = {}
NEXT_MOCK_TICKET_NUMBER = 1001
MAX_TOOL_ROUNDS = 3

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "Create an in-memory mock IT helpdesk ticket for this demo. "
                "Call only when the employee explicitly asks to create, open, file, or submit a ticket. "
                "This does not contact a real IT system."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "A concise summary of the employee's IT issue.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Ticket priority. Use medium if the employee did not specify one.",
                    },
                },
                "required": ["issue", "priority"],
                "additionalProperties": False,
            },
        },
    }
]

TICKET_REQUEST_PATTERN = re.compile(
    r"\b(create|open|file|raise|submit|log)\b.{0,60}\b(ticket|case)\b"
    r"|\b(ticket|case)\b.{0,60}\b(create|open|file|raise|submit|log)\b",
    re.IGNORECASE,
)


def create_ticket(issue: str, priority: str) -> dict[str, Any]:
    """Create an in-memory demonstration ticket; no real helpdesk is contacted."""
    global NEXT_MOCK_TICKET_NUMBER

    issue = issue.strip()
    priority = priority.strip().lower()
    if not issue:
        return {"error": "The issue summary cannot be empty."}
    if len(issue) > 500:
        return {"error": "The issue summary must be 500 characters or fewer."}
    if priority not in {"low", "medium", "high", "urgent"}:
        return {"error": "Priority must be low, medium, high, or urgent."}

    ticket_id = f"MOCK-{NEXT_MOCK_TICKET_NUMBER}"
    NEXT_MOCK_TICKET_NUMBER += 1
    ticket = {
        "ticket_id": ticket_id,
        "issue": issue,
        "priority": priority,
        "status": "mock-created",
        "note": "Demo only. This ticket was not sent to a real IT system.",
    }
    MOCK_TICKETS[ticket_id] = ticket
    return ticket


def fallback_answer(question: str) -> str:
    faq = find_local_faq(question)
    if faq:
        return str(faq.get("answer", "I could not find a matching FAQ answer."))
    return (
        "I could not find a matching approved FAQ for your question. "
        "Please contact IT through your organization's approved channel."
    )


async def send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "content": message})
    except RuntimeError:
        logger.warning("Failed to send error message after WebSocket disconnect.")


async def safe_stream_answer(
    websocket: WebSocket,
    model: str,
    messages: list[dict[str, Any]],
    question: str,
) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    ticket_requested = bool(TICKET_REQUEST_PATTERN.search(question))
    ticket_created_this_turn = False

    for _ in range(MAX_TOOL_ROUNDS):
        answer_parts: list[str] = []
        tool_deltas: dict[int, dict[str, str | None]] = {}

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                stream=True,
            )

            async for chunk in stream:
                if websocket.application_state.name != "CONNECTED":
                    raise WebSocketDisconnect()

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                token = getattr(delta, "content", None)
                if isinstance(token, str) and token:
                    answer_parts.append(token)
                    await websocket.send_json({"type": "token", "content": token})

                for call_delta in getattr(delta, "tool_calls", None) or []:
                    index = getattr(call_delta, "index", 0)
                    partial = tool_deltas.setdefault(
                        index, {"id": None, "name": "", "arguments": ""}
                    )
                    call_id = getattr(call_delta, "id", None)
                    if call_id:
                        partial["id"] = call_id
                    function_delta = getattr(call_delta, "function", None)
                    if function_delta is not None:
                        name_part = getattr(function_delta, "name", None)
                        args_part = getattr(function_delta, "arguments", None)
                        if name_part:
                            partial["name"] = (partial["name"] or "") + name_part
                        if args_part:
                            partial["arguments"] = (partial["arguments"] or "") + args_part

        except (RuntimeError, WebSocketDisconnect):
            raise
        except Exception:
            logger.exception("Groq streaming request failed")
            raise

        if tool_deltas:
            assistant_tool_calls = []
            for index, partial in sorted(tool_deltas.items()):
                assistant_tool_calls.append({
                    "id": partial["id"] or f"call-local-{index}",
                    "type": "function",
                    "function": {
                        "name": partial["name"] or "",
                        "arguments": partial["arguments"] or "{}",
                    },
                })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            })

            for tool_call in assistant_tool_calls:
                function_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
                    result: dict[str, Any] = {"error": "The model sent invalid tool arguments."}
                else:
                    if function_name != "create_ticket":
                        result = {"error": "Unknown tool."}
                    elif not ticket_requested:
                        result = {"error": "Ask the user to explicitly request a ticket before creating one."}
                    elif ticket_created_this_turn:
                        result = {"error": "Only one mock ticket can be created for this request."}
                    else:
                        issue = arguments.get("issue")
                        priority = arguments.get("priority")
                        if not isinstance(issue, str) or not isinstance(priority, str):
                            result = {"error": "Both issue and priority must be strings."}
                        else:
                            result = create_ticket(issue, priority)
                            ticket_created_this_turn = "ticket_id" in result

                await websocket.send_json({
                    "type": "tool_call",
                    "name": function_name,
                    "arguments": arguments,
                })
                await websocket.send_json({
                    "type": "tool_result",
                    "name": function_name,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": json.dumps(result),
                })
            continue

        response_text = "".join(answer_parts)
        if not response_text:
            response_text = fallback_answer(question)
            await websocket.send_json({"type": "token", "content": response_text})
        return response_text

    raise RuntimeError("The assistant exceeded the tool-call limit for one message.")


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    if client is None:
        await send_error(websocket, "Set GROQ_API_KEY in the local .env file.")
        await websocket.close(code=1011)
        return

    messages: list[dict[str, Any]] = [{"role": "system", "content": WEBSOCKET_SYSTEM_PROMPT}]

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("Helpdesk chat disconnected; in-memory history discarded.")
                return
            except Exception:
                await send_error(websocket, "Send valid JSON in the format {\"message\": \"your question\"}.")
                continue

            if not isinstance(payload, dict):
                await send_error(websocket, "Send a JSON object with a non-empty 'message' field.")
                continue

            question = payload.get("message")
            if not isinstance(question, str) or not question.strip():
                await send_error(websocket, "Send a non-empty message field.")
                continue

            question = question.strip()
            if len(question) > 2000:
                await send_error(websocket, "Keep each message under 2,000 characters.")
                continue

            faq = find_local_faq(question)
            if faq:
                user_content = (
                    f"Employee question: {question}\n\n"
                    f"Approved FAQ question: {faq['question']}\n"
                    f"Approved answer: {faq['answer']}"
                )
            else:
                user_content = f"Employee question: {question}\n\nNo matching approved FAQ was found."

            messages.append({"role": "user", "content": user_content})

            try:
                response_text = await safe_stream_answer(websocket, MODEL, messages, question)
            except WebSocketDisconnect:
                logger.info("Client disconnected while receiving response.")
                return
            except Exception:
                messages.pop()
                await send_error(websocket, "The assistant could not complete the request.")
                continue

            if response_text:
                messages.append({"role": "assistant", "content": response_text})
            else:
                fallback = fallback_answer(question)
                messages.append({"role": "assistant", "content": fallback})
                await websocket.send_json({"type": "token", "content": fallback})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("Helpdesk chat disconnected; in-memory history discarded.")

