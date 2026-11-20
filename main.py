import json, logging, os, re
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from groq import AsyncGroq
from bot import SYSTEM_PROMPT, find_local_faq

load_dotenv()
logging.basicConfig(level=logging.INFO)
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
client = AsyncGroq(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="Alish IT Helpdesk")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")
WS_PROMPT = f"{SYSTEM_PROMPT}\nAlways call search_faq before answering policy/support questions. Call create_ticket only when explicitly asked; share the mock ID and state no real IT system was contacted."

TICKETS: dict[str, dict[str, Any]] = {}
NEXT_ID = 1001
TOOLS = [
    {"type": "function", "function": {"name": "search_faq", "description": "Search local IT FAQs.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "create_ticket", "description": "Create a mock IT ticket.", "parameters": {"type": "object", "properties": {"issue": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}}, "required": ["issue", "priority"]}}},
]
TICKET_RE = re.compile(r"\b(create|open|file|raise|submit|log)\b.{0,60}\b(ticket|case)\b|\b(ticket|case)\b.{0,60}\b(create|open|file|raise|submit|log)\b", re.I)

def create_ticket(issue: str, priority: str) -> dict[str, Any]:
    global NEXT_ID
    issue, priority = issue.strip(), priority.strip().lower()
    if not issue or len(issue) > 500: return {"error": "Issue must be 1-500 chars."}
    if priority not in {"low", "medium", "high", "urgent"}: return {"error": "Invalid priority."}
    tid = f"MOCK-{NEXT_ID}"
    NEXT_ID += 1
    TICKETS[tid] = {"ticket_id": tid, "issue": issue, "priority": priority, "status": "mock-created"}
    return TICKETS[tid]

def fallback(q: str) -> str:
    f = find_local_faq(q)
    return str(f["answer"]) if f else "No matching approved FAQ found. Contact IT support."

async def send_err(ws: WebSocket, msg: str) -> None:
    try: await ws.send_json({"type": "error", "content": msg})
    except RuntimeError: pass

def bound_msgs(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not msgs: return []
    sys_m = msgs[0] if msgs[0].get("role") == "system" else None
    recent = (msgs[1:] if sys_m else msgs)[-6:]
    keep = {"role", "content", "tool_calls", "tool_call_id", "name"}
    out = [{"role": "system", "content": str(sys_m.get("content", ""))[:1200]}] if sys_m else []
    for m in recent:
        item = {k: v for k, v in m.items() if k in keep}
        if isinstance(item.get("content"), str) and len(item["content"]) > 1200:
            item["content"] = item["content"][:1200].rstrip() + "..."
        out.append(item)
    pre, hist = (out[:1], out[1:]) if sys_m else ([], out[:])
    while len(json.dumps({"messages": pre + hist, "tools": TOOLS}).encode()) > 5600:
        idx = next((i for i, m in enumerate(hist[1:], 1) if m.get("role") == "user"), None)
        if idx is None: break
        hist = hist[idx:]
    return pre + hist

def run_tool(name: str, args: dict[str, Any], req_t: bool, created: bool) -> tuple[dict[str, Any], bool]:
    if name == "search_faq":
        q = args.get("query")
        if not isinstance(q, str) or not q.strip(): return {"error": "Empty query."}, created
        f = find_local_faq(q)
        return ({"question": f["question"], "answer": f["answer"]} if f else {"answer": "No matching FAQ."}), created
    if name != "create_ticket": return {"error": "Unknown tool."}, created
    if not req_t: return {"error": "Explicit ticket request required."}, created
    if created: return {"error": "One ticket per request."}, created
    issue, prio = args.get("issue"), args.get("priority")
    if not isinstance(issue, str) or not isinstance(prio, str): return {"error": "Invalid args."}, created
    res = create_ticket(issue, prio)
    return res, created or ("ticket_id" in res)

async def stream_answer(ws: WebSocket, msgs: list[dict[str, Any]], q: str) -> str:
    if not client: raise RuntimeError("Missing API key")
    req_t, created = bool(TICKET_RE.search(q)), False
    for _ in range(3):
        parts, deltas = [], {}
        try:
            stream = await client.chat.completions.create(model=MODEL, messages=bound_msgs(msgs), tools=TOOLS, tool_choice="auto", stream=True, max_tokens=600)
            async for chunk in stream:
                d = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                if not d: continue
                if getattr(d, "content", None):
                    parts.append(d.content)
                    await ws.send_json({"type": "token", "content": d.content})
                for cd in getattr(d, "tool_calls", None) or []:
                    p = deltas.setdefault(getattr(cd, "index", 0), {"id": None, "name": "", "arguments": ""})
                    if getattr(cd, "id", None): p["id"] = cd.id
                    if getattr(cd, "function", None):
                        p["name"] += getattr(cd.function, "name", "") or ""
                        p["arguments"] += getattr(cd.function, "arguments", "") or ""
        except (RuntimeError, WebSocketDisconnect): raise
        except Exception as e:
            if "request too large" in str(e).lower() or "tokens per minute" in str(e).lower():
                txt = fallback(q)
                await ws.send_json({"type": "token", "content": txt})
                return txt
            raise
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
                if not isinstance(args, dict): raise ValueError
                res, created = run_tool(fn, args, req_t, created)
            except Exception:
                args, res = {}, {"error": "Invalid tool arguments."}
            await ws.send_json({"type": "tool_call", "name": fn, "arguments": args})
            await ws.send_json({"type": "tool_result", "name": fn, "result": res})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": fn, "content": json.dumps(res)})
    raise RuntimeError("Tool limit exceeded")

@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket) -> None:
    await ws.accept()
    if not client:
        await send_err(ws, "Set GROQ_API_KEY in .env.")
        await ws.close(code=1011)
        return
    msgs = [{"role": "system", "content": WS_PROMPT}]
    try:
        while True:
            try: payload = await ws.receive_json()
            except WebSocketDisconnect: return
            except Exception:
                await send_err(ws, 'Send JSON: {"message": "..."}')
                continue
            q = payload.get("message") if isinstance(payload, dict) else None
            if not isinstance(q, str) or not q.strip() or len(q.strip()) > 2000:
                await send_err(ws, "Send a non-empty message under 2,000 chars.")
                continue
            q = q.strip()
            msgs.append({"role": "user", "content": f"Employee question: {q}"})
            try: reply = await stream_answer(ws, msgs, q)
            except WebSocketDisconnect: return
            except Exception:
                msgs.pop()
                await send_err(ws, "Could not complete request.")
                continue
            msgs.append({"role": "assistant", "content": reply})
            await ws.send_json({"type": "done"})
    except WebSocketDisconnect: pass

@app.get("/")
async def home() -> FileResponse:
    return FileResponse(INDEX_PATH)