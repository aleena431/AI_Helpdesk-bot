"""FastAPI WebSocket chat endpoint with per-connection memory and streaming."""

import logging
import os

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

app = FastAPI(title="Alina IT Helpdesk Assistant")


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    if client is None:
        await websocket.send_json(
            {"type": "error", "content": "Set GROQ_API_KEY in the local .env file."}
        )
        await websocket.close(code=1011)
        return

    # This list belongs only to this WebSocket connection and disappears at disconnect.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    try:
        while True:
            payload = await websocket.receive_json()
            question = payload.get("message") if isinstance(payload, dict) else None
            if not isinstance(question, str) or not question.strip():
                await websocket.send_json(
                    {"type": "error", "content": "Send a non-empty message field."}
                )
                continue

            question = question.strip()
            if len(question) > 2000:
                await websocket.send_json(
                    {"type": "error", "content": "Keep each message under 2,000 characters."}
                )
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
            answer_parts: list[str] = []

            try:
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                )
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        answer_parts.append(token)
                        await websocket.send_json({"type": "token", "content": token})
            except Exception:
                messages.pop()
                logger.exception("Groq streaming request failed")
                await websocket.send_json(
                    {"type": "error", "content": "The assistant could not complete the request."}
                )
                continue

            messages.append({"role": "assistant", "content": "".join(answer_parts)})
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("Helpdesk chat disconnected; in-memory history discarded.")

