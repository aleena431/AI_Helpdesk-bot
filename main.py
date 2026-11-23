import json, os, re
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from groq import AsyncGroq
from bot import SYSTEM_PROMPT, find_local_faq

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
client = AsyncGroq(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="Alish IT Helpdesk")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")
WS_PROMPT = f"{SYSTEM_PROMPT} Always call search_faq before answering support questions. Call create_ticket only when explicitly requested, sharing the mock ID and noting no real IT system was contacted."

TICKETS: dict[str, dict[str, Any]] = {}
NEXT_ID = 1001
TOOLS = [
    {"type": "function", "function": {"name": "search_faq", "description": "Search IT FAQs.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "create_ticket", "description": "Create mock IT ticket.", "parameters": {"type": "object", "properties": {"issue": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}}, "required": ["issue", "priority"]}}},
]
TICKET_RE = re.compile(r"\b(create|open|file|raise|submit|log)\b.{0,40}\b(ticket|case)\b", re.I)

def create_ticket(issue: str, priority: str) -> dict[str, Any]:
    global NEXT_ID
    issue, priority = issue.strip(), priority.strip().lower()
    if not issue or priority not in {"low", "medium", "high", "urgent"}:
        return {"error": "Invalid issue or priority."}
    tid = f"MOCK-{NEXT_ID}"
    NEXT_ID += 1
    TICKETS[tid] = {"ticket_id": tid, "issue": issue, "priority": priority, "status": "mock-created"}
    return TICKETS[tid]

def fallback(q: str) -> str:
    f = find_local_faq(q)
    return str(f["answer"]) if f else "No matching approved FAQ found. Contact IT support."

def bound_msgs(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [msgs[0]] + msgs[1:][-6:]

def run_tool(name: str, args: dict[str, Any], req_t: bool, created: bool) -> tuple[dict[str, Any], bool]:
    if name == "search_faq":
        q = str(args.get("query", "")).strip()
        if not q: return {"error": "Empty query."}, created
        f = find_local_faq(q)
        return ({"question": f["question"], "answer": f["answer"]} if f else {"answer": "No matching FAQ."}), created
    if name != "create_ticket" or not req_t or created:
        return {"error": "Ticket creation requires explicit single request."}, created
    res = create_ticket(str(args.get("issue", "")), str(args.get("priority", "")))
    return res, "ticket_id" in res

async def stream_answer(ws: WebSocket, msgs: list[dict[str, Any]], q: str) -> str:
    req_t, created = bool(TICKET_RE.search(q)), False
    for _ in range(3):
        parts, deltas = [], {}
        stream = await client.chat.completions.create(model=MODEL, messages=bound_msgs(msgs), tools=TOOLS, tool_choice="auto", stream=True, max_tokens=500)
        async for chunk in stream:
            d = chunk.choices[0].delta if chunk.choices else None
            if not d: continue
            if d.content:
                parts.append(d.content)
                await ws.send_json({"type": "token", "content": d.content})
            for cd in d.tool_calls or []:
                p = deltas.setdefault(cd.index, {"id": None, "name": "", "arguments": ""})
                if cd.id: p["id"] = cd.id
                if cd.function:
                    p["name"] += cd.function.name or ""
                    p["arguments"] += cd.function.arguments or ""
        if not deltas:
            txt = "".join(parts) or fallback(q)
            if not parts: await ws.send_json({"type": "token", "content": txt})
            return txt
        calls = [{"id": p["id"] or f"call-{i}", "type": "function", "function": {"name": p["name"], "arguments": p["arguments"] or "{}"}} for i, p in sorted(deltas.items())]
        msgs.append({"role": "assistant", "content": None, "tool_calls": calls})
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"])
                res, created = run_tool(fn, args, req_t, created)
            except Exception:
                args, res = {}, {"error": "Invalid tool arguments."}
            await ws.send_json({"type": "tool_call", "name": fn, "arguments": args})
            await ws.send_json({"type": "tool_result", "name": fn, "result": res})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": fn, "content": json.dumps(res)})
    return fallback(q)

@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket) -> None:
    await ws.accept()
    if not client:
        await ws.send_json({"type": "error", "content": "Set GROQ_API_KEY in .env."})
        return await ws.close(code=1011)
    msgs = [{"role": "system", "content": WS_PROMPT}]
    try:
        while True:
            payload = await ws.receive_json()
            q = str(payload.get("message", "")).strip() if isinstance(payload, dict) else ""
            if not q or len(q) > 2000:
                await ws.send_json({"type": "error", "content": "Send a message under 2,000 chars."})
                continue
            msgs.append({"role": "user", "content": q})
            try:
                reply = await stream_answer(ws, msgs, q)
            except WebSocketDisconnect:
                return
            except Exception:
                msgs.pop()
                await ws.send_json({"type": "error", "content": "Request failed."})
                continue
            msgs.append({"role": "assistant", "content": reply})
            await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass

@app.get("/")
async def home() -> FileResponse:
    return FileResponse(INDEX_PATH)