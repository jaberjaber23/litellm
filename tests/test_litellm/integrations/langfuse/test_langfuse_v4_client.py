"""Covers client construction and lifecycle against the real v4 SDK.

These guard the failure modes the v2 to v4 move introduces: an unsupported SDK
dropping every event behind a swallowed import error, langfuse and litellm
fighting over one tracer provider, and a rotated credential still exporting to
the old destination.
"""

import pytest
import opentelemetry.trace as otel_trace
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import TracerProvider

from litellm.integrations.langfuse.langfuse import (
    MINIMUM_LANGFUSE_VERSION,
    _raise_if_unsupported_langfuse_version,
    installed_langfuse_version,
)
from litellm.integrations.langfuse.langfuse_v4_client import (
    build_isolated_tracer_provider,
    evict_stale_langfuse_resources,
    shutdown_langfuse_client,
)

PUBLIC_KEY = "pk-lifecycle-test"


@pytest.fixture(autouse=True)
def _clean_registry():
    LangfuseResourceManager._instances.pop(PUBLIC_KEY, None)
    yield
    LangfuseResourceManager._instances.pop(PUBLIC_KEY, None)


def _client(secret_key="sk-original", host="http://127.0.0.1:1"):
    return Langfuse(
        public_key=PUBLIC_KEY,
        secret_key=secret_key,
        host=host,
        tracer_provider=build_isolated_tracer_provider(environment=None, release=None),
    )


@pytest.mark.parametrize("unsupported", ["2.59.7", "3.15.0", "5.0.0"], ids=["v2", "v3", "v5"])
def test_unsupported_sdk_fails_loudly_rather_than_dropping_every_event(unsupported):
    with pytest.raises(ImportError) as raised:
        _raise_if_unsupported_langfuse_version(unsupported)
    assert unsupported in str(raised.value)
    assert MINIMUM_LANGFUSE_VERSION in str(raised.value)


def test_supported_sdk_is_accepted():
    assert _raise_if_unsupported_langfuse_version(installed_langfuse_version()) is None


def test_isolated_provider_carries_environment_and_release():
    provider = build_isolated_tracer_provider(environment="staging", release="v9")
    attributes = provider.resource.attributes
    assert attributes["langfuse.environment"] == "staging"
    assert attributes["langfuse.release"] == "v9"


def test_client_does_not_take_over_the_process_tracer_provider():
    # the global provider can only be set once per process, so assert it is left
    # alone rather than assuming this test is the one that installed it
    provider_before = otel_trace.get_tracer_provider()

    client = _client()

    assert otel_trace.get_tracer_provider() is provider_before
    assert client._resources.tracer_provider is not provider_before
    active = getattr(provider_before, "_active_span_processor", None)
    if active is not None:
        assert not any(
            "Langfuse" in type(processor).__name__ for processor in active._span_processors
        )


def test_rotated_credentials_replace_the_cached_client():
    original = _client(secret_key="sk-original", host="http://127.0.0.1:1")
    original_resources = original._resources

    evict_stale_langfuse_resources(
        public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2"
    )
    rotated = _client(secret_key="sk-rotated", host="http://127.0.0.1:2")

    assert rotated._resources is not original_resources
    assert rotated._resources.secret_key == "sk-rotated"
    assert rotated._resources.base_url == "http://127.0.0.1:2"


def test_unchanged_credentials_keep_the_cached_client():
    original = _client()
    evict_stale_langfuse_resources(
        public_key=PUBLIC_KEY, secret_key="sk-original", base_url="http://127.0.0.1:1"
    )
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is original._resources


def test_eviction_flushes_queued_observations_before_tearing_down():
    """An observation already ended when the cache evicts must still reach langfuse."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from litellm.integrations.langfuse.langfuse_v4_observations import (
        open_trace_context,
        start_generation,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # a long delay keeps the span queued, so only the shutdown can flush it
    provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=600000))
    client = Langfuse(
        public_key=PUBLIC_KEY,
        secret_key="sk-original",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    context, claim_root = open_trace_context(client=client, trace_id="a" * 32, parent_observation_id=None)
    start_generation(
        client=client, context=context, name="in-flight", start_time=None, claim_trace_root=claim_root, attributes={}
    ).end()
    assert exporter.get_finished_spans() == ()

    shutdown_langfuse_client(client)

    assert any(span.name == "in-flight" for span in exporter.get_finished_spans())


def test_shutdown_deregisters_so_a_later_client_is_not_a_corpse():
    client = _client()
    resources = client._resources

    shutdown_langfuse_client(client)

    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is not resources


def test_shutdown_of_a_stale_client_does_not_deregister_the_live_one():
    stale = _client(secret_key="sk-original", host="http://127.0.0.1:1")
    stale_resources = stale._resources
    evict_stale_langfuse_resources(
        public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2"
    )
    live = _client(secret_key="sk-rotated", host="http://127.0.0.1:2")

    shutdown_langfuse_client(stale)

    assert stale_resources is not live._resources
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is live._resources
