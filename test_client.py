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
    async with websockets.connect(WS_URL) as websocket:
        for turn_number, question in enumerate(TURNS, start=1):
            print(f"\nYou ({turn_number}/3): {question}")
            await websocket.send(json.dumps({"message": question}))
            print("Alina: ", end="", flush=True)

            while True:
                event = json.loads(await websocket.recv())
                if event["type"] == "token":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "done":
                    print()
                    break
                elif event["type"] == "error":
                    print(f"\nError: {event['content']}")
                    return


if __name__ == "__main__":
    asyncio.run(main())
