# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the VibeVoice same-origin browser proxy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

APP_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/text_to_speech/vibevoice/app.py"
spec = importlib.util.spec_from_file_location("vibevoice_browser_example_test", APP_PATH)
assert spec is not None and spec.loader is not None
browser_example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = browser_example
spec.loader.exec_module(browser_example)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_browser_proxy_serves_player_and_worklet() -> None:
    client = TestClient(browser_example.build_app(upstream="http://127.0.0.1:8000", timeout=1.0))

    player = client.get("/player")
    worklet = client.get("/pcm-player-worklet.js")

    assert player.status_code == 200
    assert "speech.audio.error" in player.text
    assert "StreamingLinearResampler" in player.text
    assert worklet.status_code == 200
    assert "registerProcessor" in worklet.text


def test_browser_proxy_closes_client_when_upstream_connection_fails(monkeypatch) -> None:
    instances = []

    class FailingAsyncClient:
        def __init__(self, **_kwargs) -> None:
            self.closed = False
            instances.append(self)

        def build_request(self, method: str, url: str, *, json: dict) -> httpx.Request:
            return httpx.Request(method, url, json=json)

        async def send(self, request: httpx.Request, *, stream: bool):
            assert stream is True
            raise httpx.ConnectError("connection refused", request=request)

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(browser_example.httpx, "AsyncClient", FailingAsyncClient)
    client = TestClient(browser_example.build_app(upstream="http://127.0.0.1:1", timeout=1.0))

    response = client.post("/proxy/speech", json={"input": "hello"})

    assert response.status_code == 502
    assert "connection refused" in response.text
    assert len(instances) == 1
    assert instances[0].closed is True
