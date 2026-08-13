"""Covers the v4 observation plumbing: historical timestamps and id normalisation.

The timestamp assertions are the regression guard for the migration: v4 has no
public API for an observation start time, so a callback running after the model
call would otherwise record its own duration instead of the call's.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
import opentelemetry.trace as otel_trace
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.langfuse.langfuse import (
    MINIMUM_LANGFUSE_VERSION,
    _raise_if_unsupported_langfuse_version,
    installed_langfuse_version,
)
from litellm.integrations.langfuse.langfuse_sdk import (
    AS_ROOT_ATTRIBUTE,
    build_isolated_tracer_provider,
    evict_stale_langfuse_resources,
    shutdown_langfuse_client,
    RELEASE_ATTRIBUTE,
    open_trace_context,
    resolve_observation_id,
    resolve_trace_id,
    start_child_span,
    start_generation,
    to_unix_nanos,
)

CALL_START = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
FIRST_TOKEN = CALL_START + timedelta(seconds=5)
CALL_END = CALL_START + timedelta(seconds=20)


@pytest.fixture(name="client")
def _client():
    LangfuseResourceManager._instances.pop("pk-obs-test", None)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key="pk-obs-test",
        secret_key="sk-obs-test",
        host="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    yield client, exporter
    LangfuseResourceManager._instances.pop("pk-obs-test", None)


def _only_span(exporter, name):
    return next(s for s in exporter.get_finished_spans() if s.name == name)


def test_generation_records_the_model_call_window_not_the_callback(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="a" * 32, parent_observation_id=None)
    start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        attributes={"completion_start_time": FIRST_TOKEN},
    ).end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    span = _only_span(exporter, "gen")
    assert span.start_time == to_unix_nanos(CALL_START)
    assert span.end_time == to_unix_nanos(CALL_END)
    assert (span.end_time - span.start_time) == 20 * 1_000_000_000
    assert json.loads(span.attributes["langfuse.observation.completion_start_time"]) == FIRST_TOKEN.isoformat().replace(
        "+00:00", "Z"
    )


def test_generation_claims_trace_root_only_without_a_real_parent(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="b" * 32, parent_observation_id=None)
    assert claim_root is True
    start_generation(
        client=lf, context=context, name="root-gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    ).end()

    parented_context, parented_claim = open_trace_context(client=lf, trace_id="b" * 32, parent_observation_id="c" * 16)
    assert parented_claim is False
    start_generation(
        client=lf,
        context=parented_context,
        name="child-gen",
        start_time=CALL_START,
        claim_trace_root=parented_claim,
        attributes={},
    ).end()
    lf.flush()

    assert _only_span(exporter, "root-gen").attributes.get(AS_ROOT_ATTRIBUTE) is True
    assert _only_span(exporter, "child-gen").attributes.get(AS_ROOT_ATTRIBUTE) is None


def test_child_span_keeps_its_own_window_and_stays_a_sibling(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="d" * 32, parent_observation_id=None)
    guardrail_start = CALL_START + timedelta(seconds=1)
    start_child_span(
        client=lf, context=context, name="guardrail", start_time=guardrail_start, attributes={}
    ).end(end_time=to_unix_nanos(guardrail_start + timedelta(seconds=2)))
    start_generation(
        client=lf, context=context, name="gen", start_time=CALL_START, claim_trace_root=claim_root, attributes={}
    ).end(end_time=to_unix_nanos(CALL_END))
    lf.flush()

    guardrail = _only_span(exporter, "guardrail")
    generation = _only_span(exporter, "gen")
    assert (guardrail.end_time - guardrail.start_time) == 2 * 1_000_000_000
    assert guardrail.parent.span_id == generation.parent.span_id
    assert guardrail.context.trace_id == generation.context.trace_id


def test_release_is_carried_on_the_root_observation(client):
    lf, exporter = client
    context, claim_root = open_trace_context(client=lf, trace_id="e" * 32, parent_observation_id=None)
    start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        release="v1.2.3",
        attributes={},
    ).end()
    lf.flush()
    assert _only_span(exporter, "gen").attributes[RELEASE_ATTRIBUTE] == "v1.2.3"


def test_request_release_beats_the_client_wide_release(monkeypatch):
    """A client configured with its own release must not overwrite trace_release."""
    monkeypatch.setenv("LANGFUSE_RELEASE", "client-wide-release")
    LangfuseResourceManager._instances.pop("pk-release-test", None)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    lf = Langfuse(
        public_key="pk-release-test",
        secret_key="sk-release-test",
        host="http://127.0.0.1:1",
        release="client-wide-release",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    context, claim_root = open_trace_context(client=lf, trace_id="f" * 32, parent_observation_id=None)
    start_generation(
        client=lf,
        context=context,
        name="gen",
        start_time=CALL_START,
        claim_trace_root=claim_root,
        release="per-request-release",
        attributes={},
    ).end()
    lf.flush()
    LangfuseResourceManager._instances.pop("pk-release-test", None)

    assert _only_span(exporter, "gen").attributes[RELEASE_ATTRIBUTE] == "per-request-release"


@pytest.mark.parametrize(
    "supplied, expected",
    [
        ("0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef"),
        ("0123456789ABCDEF0123456789ABCDEF", "0123456789abcdef0123456789abcdef"),
        ("3fe0c940-b69a-de3b-a77c-06102505349a", "3fe0c940b69ade3ba77c06102505349a"),
    ],
    ids=["already-hex", "uppercase-hex", "uuid-with-dashes"],
)
def test_trace_id_passes_through_when_it_is_already_usable(supplied, expected):
    assert resolve_trace_id(supplied) == expected


def test_arbitrary_trace_id_is_hashed_deterministically():
    first = resolve_trace_id("order-4471")
    assert first == resolve_trace_id("order-4471")
    assert len(first) == 32 and first == first.lower()
    assert first != resolve_trace_id("order-4472")


def test_missing_trace_id_still_yields_a_valid_trace_id():
    generated = resolve_trace_id(None)
    assert len(generated) == 32
    assert int(generated, 16) >= 0


@pytest.mark.parametrize(
    "supplied, expected",
    [
        ("0123456789abcdef", "0123456789abcdef"),
        (None, None),
        ("", None),
    ],
    ids=["already-hex", "none", "empty"],
)
def test_observation_id_normalisation(supplied, expected):
    assert resolve_observation_id(supplied) == expected


def test_arbitrary_observation_id_is_hashed_to_a_span_id():
    resolved = resolve_observation_id("my-parent-observation")
    assert len(resolved) == 16
    assert resolved == resolve_observation_id("my-parent-observation")


PUBLIC_KEY = "pk-lifecycle-test"


@pytest.fixture(autouse=True)
def _clean_registry():
    LangfuseResourceManager._instances.pop(PUBLIC_KEY, None)
    yield
    LangfuseResourceManager._instances.pop(PUBLIC_KEY, None)


def _lifecycle_client(secret_key="sk-original", host="http://127.0.0.1:1"):
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

    client = _lifecycle_client()

    assert otel_trace.get_tracer_provider() is provider_before
    assert client._resources.tracer_provider is not provider_before
    active = getattr(provider_before, "_active_span_processor", None)
    if active is not None:
        assert not any(
            "Langfuse" in type(processor).__name__ for processor in active._span_processors
        )


def test_rotated_credentials_replace_the_cached_client():
    original = _lifecycle_client(secret_key="sk-original", host="http://127.0.0.1:1")
    original_resources = original._resources

    evict_stale_langfuse_resources(
        public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2"
    )
    rotated = _lifecycle_client(secret_key="sk-rotated", host="http://127.0.0.1:2")

    assert rotated._resources is not original_resources
    assert rotated._resources.secret_key == "sk-rotated"
    assert rotated._resources.base_url == "http://127.0.0.1:2"


def test_unchanged_credentials_keep_the_cached_client():
    original = _lifecycle_client()
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

    from litellm.integrations.langfuse.langfuse_sdk import (
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
    client = _lifecycle_client()
    resources = client._resources

    shutdown_langfuse_client(client)

    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is not resources


def test_shutdown_of_a_stale_client_does_not_deregister_the_live_one():
    stale = _lifecycle_client(secret_key="sk-original", host="http://127.0.0.1:1")
    stale_resources = stale._resources
    evict_stale_langfuse_resources(
        public_key=PUBLIC_KEY, secret_key="sk-rotated", base_url="http://127.0.0.1:2"
    )
    live = _lifecycle_client(secret_key="sk-rotated", host="http://127.0.0.1:2")

    shutdown_langfuse_client(stale)

    assert stale_resources is not live._resources
    assert LangfuseResourceManager._instances.get(PUBLIC_KEY) is live._resources
