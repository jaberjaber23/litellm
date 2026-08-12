"""Tests for the Slack alerting model deprecation hook."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import AlertType


def _make_router(deployments):
    router = MagicMock()
    router.get_model_list.return_value = deployments
    return router


@pytest.mark.asyncio
async def test_should_skip_when_alert_type_disabled():
    alerting = SlackAlerting(
        alerting=["slack"],
        alert_types=[AlertType.llm_exceptions],
    )
    sent = await alerting.send_model_deprecation_alert(llm_router=MagicMock())
    assert sent is False


@pytest.mark.asyncio
async def test_should_skip_when_no_alerting_configured():
    alerting = SlackAlerting(
        alerting=None,
        alert_types=[AlertType.model_deprecation_warnings],
    )
    sent = await alerting.send_model_deprecation_alert(llm_router=MagicMock())
    assert sent is False


@pytest.mark.asyncio
async def test_should_skip_when_no_deprecations_found(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {})
    alerting = SlackAlerting(
        alerting=["slack"],
        alert_types=[AlertType.model_deprecation_warnings],
    )
    router = _make_router(
        [
            {
                "model_name": "fresh",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "x"},
            }
        ]
    )
    sent = await alerting.send_model_deprecation_alert(llm_router=router)
    assert sent is False


@pytest.mark.asyncio
async def test_should_dispatch_high_severity_when_deprecated(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "dead-model": {
                "deprecation_date": "2020-01-01",
                "litellm_provider": "openai",
            }
        },
    )
    alerting = SlackAlerting(
        alerting=["slack"],
        alert_types=[AlertType.model_deprecation_warnings],
    )
    router = _make_router(
        [
            {
                "model_name": "dead-alias",
                "litellm_params": {"model": "dead-model"},
                "model_info": {"id": "1"},
            }
        ]
    )

    with patch.object(
        alerting, "send_alert", new_callable=AsyncMock
    ) as mock_send_alert:
        sent = await alerting.send_model_deprecation_alert(llm_router=router)

    assert sent is True
    mock_send_alert.assert_awaited_once()
    call_kwargs = mock_send_alert.await_args.kwargs
    assert call_kwargs["alert_type"] == AlertType.model_deprecation_warnings
    assert call_kwargs["level"] == "High"
    assert call_kwargs["alerting_metadata"]["deprecated_count"] == 1
    assert call_kwargs["alerting_metadata"]["imminent_count"] == 0
    assert "dead-alias" in call_kwargs["message"]


@pytest.mark.asyncio
async def test_should_short_poll_router_before_daily_sleep(monkeypatch):
    """Regression: config load can start this loop before the proxy assigns `llm_router`.

    If the first pass ran against a None router, the loop would send no alert and then
    sleep the full 24h interval, silently postponing the startup deprecation alert.
    The short poll ensures the first real alert lands as soon as the router shows up.
    """
    from litellm.types.proxy.model_deprecation import (
        DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
        DEFAULT_DEPRECATION_ROUTER_WAIT_SECONDS,
    )

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}},
    )
    alerting = SlackAlerting(
        alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings]
    )
    router = _make_router(
        [
            {
                "model_name": "dead-alias",
                "litellm_params": {"model": "dead-model"},
                "model_info": {"id": "1"},
            }
        ]
    )
    routers = [None, None, router]
    sleeps: list[float] = []

    async def record_and_stop_after_alert(seconds):
        sleeps.append(seconds)
        if seconds == DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS:
            raise asyncio.CancelledError

    with (
        patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=record_and_stop_after_alert,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting._run_scheduled_deprecation_check(
            get_llm_router=lambda: routers.pop(0)
        )

    assert sleeps == [
        DEFAULT_DEPRECATION_ROUTER_WAIT_SECONDS,
        DEFAULT_DEPRECATION_ROUTER_WAIT_SECONDS,
        DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    ]
    mock_send_alert.assert_awaited_once()
    assert "dead-alias" in mock_send_alert.await_args.kwargs["message"]


@pytest.mark.asyncio
async def test_should_alert_once_the_alert_type_and_router_arrive_after_startup(
    monkeypatch,
):
    """The daily loop starts before config reload, so it must re-read both each pass"""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}},
    )
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.llm_exceptions])
    router = _make_router(
        [
            {
                "model_name": "dead-alias",
                "litellm_params": {"model": "dead-model"},
                "model_info": {"id": "1"},
            }
        ]
    )
    routers = [None, router]

    async def stop_after_second_pass(_seconds):
        if alerting.alert_types == [AlertType.llm_exceptions]:
            alerting.update_values(
                alert_types=[AlertType.model_deprecation_warnings]
            )  # simulates a config reload enabling the alert
            return
        raise asyncio.CancelledError

    with (
        patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=stop_after_second_pass,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting._run_scheduled_deprecation_check(
            get_llm_router=lambda: routers.pop(0)
        )

    mock_send_alert.assert_awaited_once()
    assert "dead-alias" in mock_send_alert.await_args.kwargs["message"]
