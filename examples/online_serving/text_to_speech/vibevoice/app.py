# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-origin browser proxy for the VibeVoice AudioWorklet example."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from pathlib import Path

import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

_EXAMPLE_DIR = Path(__file__).resolve().parent


def build_app(*, upstream: str, timeout: float) -> FastAPI:
    app = FastAPI()
    speech_url = f"{upstream.rstrip('/')}/v1/audio/speech"

    @app.get("/player")
    async def player() -> FileResponse:
        return FileResponse(_EXAMPLE_DIR / "player.html")

    @app.get("/pcm-player-worklet.js")
    async def worklet() -> FileResponse:
        return FileResponse(
            _EXAMPLE_DIR / "pcm-player-worklet.js",
            media_type="text/javascript",
        )

    @app.post("/proxy/speech")
    async def proxy_speech(request: Request) -> Response:
        payload = await request.json()
        client = httpx.AsyncClient(trust_env=False, timeout=timeout)
        upstream_response: httpx.Response | None = None
        handed_off = False
        try:
            upstream_request = client.build_request("POST", speech_url, json=payload)
            upstream_response = await client.send(upstream_request, stream=True)
            if upstream_response.status_code >= 400:
                content = await upstream_response.aread()
                return Response(
                    content=content,
                    status_code=upstream_response.status_code,
                    media_type=upstream_response.headers.get("content-type"),
                )

            streaming_upstream = upstream_response

            async def body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in streaming_upstream.aiter_raw():
                        yield chunk
                finally:
                    await streaming_upstream.aclose()
                    await client.aclose()

            response = StreamingResponse(
                body(),
                media_type=streaming_upstream.headers.get("content-type", "application/octet-stream"),
                headers={"Cache-Control": "no-store"},
            )
            handed_off = True
            return response
        except httpx.HTTPError as exc:
            return Response(
                content=f"Upstream speech request failed: {exc}",
                status_code=502,
                media_type="text/plain",
            )
        finally:
            if not handed_off:
                if upstream_response is not None:
                    await upstream_response.aclose()
                await client.aclose()

    with gr.Blocks(title="VibeVoice streaming AudioWorklet") as demo:
        gr.Markdown(
            "# VibeVoice streaming AudioWorklet\n"
            "The player below proxies raw PCM or SSE through this same-origin app. "
            "Open `/player` in four tabs for the concurrency/isolation check."
        )
        gr.HTML('<iframe src="/player" style="width:100%;height:760px;border:0"></iframe>')

    return gr.mount_gradio_app(app, demo, path="/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        build_app(upstream=args.upstream, timeout=args.timeout),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
