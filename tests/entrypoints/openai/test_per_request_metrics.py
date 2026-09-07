# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from upstream PR #46768 (v0.25.0): per-request timing metrics.

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    PerRequestTimingMetrics,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    build_per_request_timing_metrics,
)
from vllm.v1.metrics.stats import RequestStateStats


def _metrics() -> RequestStateStats:
    return RequestStateStats(
        queued_ts=1.0,
        scheduled_ts=1.5,
        first_token_ts=2.0,
        last_token_ts=3.0,
        num_generation_tokens=10,
    )


def test_build_per_request_timing_metrics():
    metrics = build_per_request_timing_metrics(_metrics(), num_generation_tokens=10)
    assert metrics.time_to_first_token_ms == pytest.approx(500.0)
    assert metrics.generation_time_ms == pytest.approx(1000.0)
    assert metrics.queue_time_ms == pytest.approx(500.0)
    assert metrics.mean_itl_ms == pytest.approx(1000.0 / 9)
    assert metrics.tokens_per_second == pytest.approx(10.0 / 1.5)


def test_build_per_request_timing_metrics_no_metrics():
    metrics = build_per_request_timing_metrics(None, num_generation_tokens=10)
    assert metrics.time_to_first_token_ms is None
    assert metrics.generation_time_ms is None
    assert metrics.queue_time_ms is None
    assert metrics.mean_itl_ms is None
    assert metrics.tokens_per_second is None


def test_build_per_request_timing_metrics_zero_tokens():
    # With all timestamps available but no generated tokens, mean ITL is
    # undefined (num_generation_tokens > 1 required) while tokens_per_second
    # evaluates to 0.0 since the inference interval is positive.
    metrics = build_per_request_timing_metrics(_metrics(), num_generation_tokens=0)
    assert metrics.time_to_first_token_ms == pytest.approx(500.0)
    assert metrics.generation_time_ms == pytest.approx(1000.0)
    assert metrics.queue_time_ms == pytest.approx(500.0)
    assert metrics.mean_itl_ms is None
    assert metrics.tokens_per_second == pytest.approx(0.0)


def test_build_per_request_timing_metrics_missing_timestamps():
    # Local timestamps default to 0.0 when unavailable, so every field stays
    # None (>= all guards are on zero-valued sentinels).
    metrics = build_per_request_timing_metrics(
        RequestStateStats(), num_generation_tokens=10
    )
    assert metrics.time_to_first_token_ms is None
    assert metrics.generation_time_ms is None
    assert metrics.queue_time_ms is None
    assert metrics.mean_itl_ms is None
    assert metrics.tokens_per_second is None


def test_per_request_metrics_serialization_chat():
    usage = UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    per_request_metrics = build_per_request_timing_metrics(
        _metrics(), num_generation_tokens=10
    )

    # Non-streaming: the router serializes with model_dump() (no exclusion),
    # so metrics=None is emitted as null, and populated metrics appear as a
    # full subtree.
    response = ChatCompletionResponse(
        id="chatcmpl-1",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=ChatMessage(role="assistant", content="hi")
            )
        ],
        usage=usage,
    )
    data = response.model_dump()
    assert data["metrics"] is None

    response = ChatCompletionResponse(
        id="chatcmpl-1",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=ChatMessage(role="assistant", content="hi")
            )
        ],
        usage=usage,
        metrics=per_request_metrics,
    )
    data = response.model_dump()
    assert data["metrics"] == {
        "time_to_first_token_ms": 500.0,
        "generation_time_ms": 1000.0,
        "queue_time_ms": 500.0,
        "mean_itl_ms": pytest.approx(1000.0 / 9),
        "tokens_per_second": pytest.approx(10.0 / 1.5),
    }

    # Streaming final usage chunk: serialized with exclude_unset=True and
    # exclude_none=True, so metrics=None is omitted entirely.
    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-1", model="test-model", choices=[], usage=usage
    )
    assert "metrics" not in json.loads(
        chunk.model_dump_json(exclude_unset=True, exclude_none=True)
    )

    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-1",
        model="test-model",
        choices=[],
        usage=usage,
        metrics=per_request_metrics,
    )
    assert "metrics" in json.loads(
        chunk.model_dump_json(exclude_unset=True, exclude_none=True)
    )


def test_per_request_metrics_serialization_completion():
    usage = UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    per_request_metrics = build_per_request_timing_metrics(
        _metrics(), num_generation_tokens=10
    )

    response = CompletionResponse(
        id="cmpl-1",
        model="test-model",
        choices=[CompletionResponseChoice(index=0, text="hi")],
        usage=usage,
    )
    assert response.model_dump()["metrics"] is None

    response = CompletionResponse(
        id="cmpl-1",
        model="test-model",
        choices=[CompletionResponseChoice(index=0, text="hi")],
        usage=usage,
        metrics=per_request_metrics,
    )
    data = response.model_dump()
    assert data["metrics"] == {
        "time_to_first_token_ms": 500.0,
        "generation_time_ms": 1000.0,
        "queue_time_ms": 500.0,
        "mean_itl_ms": pytest.approx(1000.0 / 9),
        "tokens_per_second": pytest.approx(10.0 / 1.5),
    }

    chunk = CompletionStreamResponse(
        id="cmpl-1", model="test-model", choices=[], usage=usage
    )
    assert "metrics" not in json.loads(
        chunk.model_dump_json(exclude_unset=False, exclude_none=True)
    )

    chunk = CompletionStreamResponse(
        id="cmpl-1",
        model="test-model",
        choices=[],
        usage=usage,
        metrics=per_request_metrics,
    )
    assert "metrics" in json.loads(
        chunk.model_dump_json(exclude_unset=False, exclude_none=True)
    )
