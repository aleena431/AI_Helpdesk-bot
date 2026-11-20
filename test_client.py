"""Run three streamed turns against the local /ws/chat endpoint."""

import asyncio
import json
import sys

import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WS_URL = "ws://127.0.0.1:8000/ws/chat"
TURNS = [
    "Search the approved FAQ: Can I share my password with the helpdesk so they can fix my account?",
    "My company VPN keeps disconnecting after I restart the approved client. Please create a high-priority ticket.",
    "What issue was in the ticket you just created?",
]


async def main() -> None:
    turns = TURNS[:1] if "--faq-only" in sys.argv else TURNS
    try:
        async with websockets.connect(WS_URL) as websocket:
            for turn_number, question in enumerate(turns, start=1):
                print(f"\nYou ({turn_number}/{len(turns)}): {question}")
                await websocket.send(json.dumps({"message": question}))
                print("Alish: ", end="", flush=True)

                while True:
                    try:
                        event = json.loads(await websocket.recv())
                    except websockets.ConnectionClosed:
                        print("\nConnection closed by server.")
                        return
                    except (json.JSONDecodeError, TypeError):
                        print("\nReceived malformed response from server.")
                        return

                    event_type = event.get("type")
                    if event_type == "token":
                        print(event.get("content", ""), end="", flush=True)
                    elif event_type == "tool_call":
                        arguments = json.dumps(event.get("arguments", {}), ensure_ascii=False)
                        print(f"\n[Tool call] {event.get('name')}({arguments})")
                    elif event_type == "tool_result":
                        result = json.dumps(event.get("result", {}), ensure_ascii=False)
                        print(f"[Tool result] {result}")
                    elif event_type == "done":
                        print()
                        break
                    elif event_type == "error":
                        print(f"\nError: {event.get('content', 'Unknown server error')}")
                        return
                    else:
                        print(f"\nUnexpected event: {event}")
                        return
    except (ConnectionRefusedError, OSError, websockets.WebSocketException) as exc:
        print(f"Unable to connect to {WS_URL}: {exc}")
        print("Start the FastAPI app first with: uvicorn main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
