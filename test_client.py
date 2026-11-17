"""Run three streamed turns against the local /ws/chat endpoint."""

import asyncio
import json

import websockets

WS_URL = "ws://127.0.0.1:8000/ws/chat"
TURNS = [
    "My VPN will not connect. What should I try?",
    "I tried restarting it, but it is still failing.",
    "What details should I send IT about this problem?",
]


async def main() -> None:
    try:
        async with websockets.connect(WS_URL) as websocket:
            for turn_number, question in enumerate(TURNS, start=1):
                print(f"\nYou ({turn_number}/3): {question}")
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
