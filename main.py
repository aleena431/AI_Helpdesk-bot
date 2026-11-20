"""FastAPI WebSocket chat endpoint with per-connection memory and streaming."""

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from groq import AsyncGroq

from bot import SYSTEM_PROMPT, find_local_faq

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
client = AsyncGroq(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="Alish IT Helpdesk Assistant")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")

WEBSOCKET_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\nFor every policy or procedure question, call search_faq before answering, even if you think you know the answer. Base the reply on the returned approved FAQ."
    + "\ncreate_ticket creates an in-memory mock ticket only when explicitly requested. "
    "After creating one, share the mock ID and state that no real IT system was contacted."
)

MOCK_TICKETS: dict[str, dict[str, Any]] = {}
NEXT_MOCK_TICKET_NUMBER = 1001
MAX_TOOL_ROUNDS = 3
MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_CHARS = 1200
# Leave room beneath the 7,000 TPM target for request framing and the response.
MAX_PROMPT_BYTES = 5600
MAX_COMPLETION_TOKENS = 600

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search approved local IT FAQs before answering policy or support questions.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Employee question."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a demo mock ticket only when explicitly asked to create or file a ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string", "description": "Concise summary of the IT issue."},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Ticket priority (default medium).",
                    },
                },
                "required": ["issue", "priority"],
                "additionalProperties": False,
            },
        },
    },
]

TICKET_RE = re.compile(
    r"\b(create|open|file|raise|submit|log)\b.{0,60}\b(ticket|case)\b"
    r"|\b(ticket|case)\b.{0,60}\b(create|open|file|raise|submit|log)\b",
    re.IGNORECASE,
)


def create_ticket(issue: str, priority: str) -> dict[str, Any]:
    global NEXT_MOCK_TICKET_NUMBER
    issue, priority = issue.strip(), priority.strip().lower()
    if not issue or len(issue) > 500:
        return {"error": "Issue must be between 1 and 500 characters."}
    if priority not in {"low", "medium", "high", "urgent"}:
        return {"error": "Priority must be low, medium, high, or urgent."}

    ticket_id = f"MOCK-{NEXT_MOCK_TICKET_NUMBER}"
    NEXT_MOCK_TICKET_NUMBER += 1
    ticket = {
        "ticket_id": ticket_id,
        "issue": issue,
        "priority": priority,
        "status": "mock-created",
        "note": "Demo only; not sent to a real IT system.",
    }
    MOCK_TICKETS[ticket_id] = ticket
    return ticket


def fallback_answer(question: str) -> str:
    faq = find_local_faq(question)
    if faq:
        return str(faq.get("answer", "No matching FAQ answer found."))
    return "No matching approved FAQ found. Contact IT through an approved channel."


async def send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "content": message})
    except RuntimeError:
        logger.warning("Failed to send error after WebSocket disconnect.")


def bounded_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return []
    system = messages[0] if messages[0].get("role") == "system" else None
    recent = (messages[1:] if system else messages)[-MAX_HISTORY_MESSAGES:]
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": str(system.get("content", ""))[:MAX_MESSAGE_CHARS]})
    for msg in recent:
        content = msg.get("content")
        if isinstance(content, str) and len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS].rstrip() + "..."
        item = {k: v for k, v in msg.items() if k in allowed}
        item["content"] = content
        out.append(item)

    def serialized_size(items: list[dict[str, Any]]) -> int:
        request = json.dumps(
            {"messages": items, "tools": TOOL_SCHEMAS}, ensure_ascii=False
        )
        return len(request.encode("utf-8"))

    # UTF-8 bytes are a conservative upper bound for text-token count. Drop
    # complete older user turns while preserving the system prompt and current turn.
    history = out[1:] if system else out[:]
    prefix = out[:1] if system else []
    while serialized_size(prefix + history) > MAX_PROMPT_BYTES:
        later_user = next(
            (i for i, msg in enumerate(history[1:], start=1) if msg.get("role") == "user"),
            None,
        )
        if later_user is None:
            break
        history = history[later_user:]
    out = prefix + history
    return out


def run_tool(
    name: str, args: dict[str, Any], ticket_requested: bool, ticket_created: bool
) -> tuple[dict[str, Any], bool]:
    if name == "search_faq":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"error": "A non-empty query is required."}, ticket_created
        faq = find_local_faq(query)
        return (
            ({"question": faq["question"], "answer": faq["answer"]} if faq else {"answer": "No matching FAQ found."}),
            ticket_created,
        )
    if name != "create_ticket":
        return {"error": "Unknown tool."}, ticket_created
    if not ticket_requested:
        return {"error": "Ask the user to explicitly request a ticket first."}, ticket_created
    if ticket_created:
        return {"error": "Only one mock ticket can be created per request."}, ticket_created

    issue, priority = args.get("issue"), args.get("priority")
    if not isinstance(issue, str) or not isinstance(priority, str):
        return {"error": "Both issue and priority must be strings."}, ticket_created
    res = create_ticket(issue, priority)
    return res, ticket_created or ("ticket_id" in res)


async def safe_stream_answer(
    websocket: WebSocket, model: str, messages: list[dict[str, Any]], question: str
) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    ticket_requested = bool(TICKET_RE.search(question))
    ticket_created = False

    for _ in range(MAX_TOOL_ROUNDS):
        answer_parts: list[str] = []
        tool_deltas: dict[int, dict[str, str | None]] = {}

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=bounded_messages(messages),
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                stream=True,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                delta = getattr(choices[0], "delta", None) if choices else None
                if delta is None:
                    continue
                token = getattr(delta, "content", None)
                if isinstance(token, str) and token:
                    answer_parts.append(token)
                    await websocket.send_json({"type": "token", "content": token})
                for call_delta in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(call_delta, "index", 0)
                    partial = tool_deltas.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                    if getattr(call_delta, "id", None):
                        partial["id"] = call_delta.id
                    fn = getattr(call_delta, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            partial["name"] = (partial["name"] or "") + fn.name
                        if getattr(fn, "arguments", None):
                            partial["arguments"] = (partial["arguments"] or "") + fn.arguments
        except (RuntimeError, WebSocketDisconnect):
            raise
        except Exception as exc:
            err = str(exc).lower()
            if "request too large" in err or "tokens per minute" in err:
                logger.warning("Groq TPM limit hit; using local FAQ fallback.")
                text = fallback_answer(question)
                await websocket.send_json({"type": "token", "content": text})
                return text
            logger.exception("Groq streaming request failed")
            raise

        if not tool_deltas:
            text = "".join(answer_parts)
            if not text:
                text = fallback_answer(question)
                await websocket.send_json({"type": "token", "content": text})
            return text

        calls = [
            {
                "id": p["id"] or f"call-local-{i}",
                "type": "function",
                "function": {"name": p["name"] or "", "arguments": p["arguments"] or "{}"},
            }
            for i, p in sorted(tool_deltas.items())
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})

        for call in calls:
            fn_name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
                if not isinstance(args, dict):
                    raise ValueError("Arguments must be a JSON object.")
                result, ticket_created = run_tool(fn_name, args, ticket_requested, ticket_created)
            except (json.JSONDecodeError, ValueError):
                args, result = {}, {"error": "Invalid tool arguments."}

            await websocket.send_json({"type": "tool_call", "name": fn_name, "arguments": args})
            await websocket.send_json({"type": "tool_result", "name": fn_name, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": fn_name,
                "content": json.dumps(result),
            })

    raise RuntimeError("Exceeded tool-call limit for one message.")


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    if client is None:
        await send_error(websocket, "Set GROQ_API_KEY in .env.")
        await websocket.close(code=1011)
        return

    messages: list[dict[str, Any]] = [{"role": "system", "content": WEBSOCKET_SYSTEM_PROMPT}]
    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                return
            except Exception:
                await send_error(websocket, 'Send valid JSON: {"message": "your question"}.')
                continue

            question = payload.get("message") if isinstance(payload, dict) else None
            if not isinstance(question, str) or not question.strip():
                await send_error(websocket, "Send a non-empty 'message' field.")
                continue

            question = question.strip()
            if len(question) > 2000:
                await send_error(websocket, "Keep each message under 2,000 characters.")
                continue

            user_content = f"Employee question: {question}"
            messages.append({"role": "user", "content": user_content})

            try:
                response_text = await safe_stream_answer(websocket, MODEL, messages, question)
            except WebSocketDisconnect:
                return
            except Exception:
                messages.pop()
                await send_error(websocket, "Could not complete the request.")
                continue

            messages.append({"role": "assistant", "content": response_text})
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.info("Helpdesk chat disconnected.")


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(INDEX_PATH)
