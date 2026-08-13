from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from typing import Final

import opentelemetry.trace as otel_trace
from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan, propagate_attributes
from opentelemetry.context import Context

__all__ = (
    "AS_ROOT_ATTRIBUTE",
    "RELEASE_ATTRIBUTE",
    "open_trace_context",
    "propagate_attributes",
    "resolve_observation_id",
    "resolve_trace_id",
    "start_child_span",
    "start_generation",
    "to_unix_nanos",
)

AS_ROOT_ATTRIBUTE: Final = "langfuse.internal.as_root"
RELEASE_ATTRIBUTE: Final = "langfuse.release"
_TRACE_ID_PATTERN: Final = re.compile(r"^[0-9a-f]{32}$")
_OBSERVATION_ID_PATTERN: Final = re.compile(r"^[0-9a-f]{16}$")


def to_unix_nanos(value: datetime | None) -> int | None:
    """Langfuse v4 takes OTel timestamps, which are integer nanoseconds since the epoch."""
    if value is None:
        return None
    return int(value.timestamp() * 1_000_000_000)


def resolve_trace_id(trace_id: str | None) -> str:
    """Map litellm's trace id onto the 32 lowercase hex characters v4 requires.

    Anything else raises inside the SDK rather than being ignored, so a plain
    uuid is dash-stripped and any other identifier is hashed deterministically,
    which keeps repeat calls with the same id on the same trace.
    """
    normalized: Final = trace_id.lower().replace("-", "") if trace_id else ""
    if _TRACE_ID_PATTERN.match(normalized):
        return normalized
    return Langfuse.create_trace_id(seed=trace_id) if trace_id else Langfuse.create_trace_id()


def resolve_observation_id(observation_id: str | None) -> str | None:
    """Same for a caller-supplied parent, which v4 requires to be 16 hex characters."""
    normalized: Final = observation_id.lower().replace("-", "") if observation_id else ""
    if not normalized:
        return None
    if _OBSERVATION_ID_PATTERN.match(normalized):
        return normalized
    return sha256(normalized.encode("utf-8")).digest()[:8].hex()


def open_trace_context(
    *,
    client: Langfuse,
    trace_id: str,
    parent_observation_id: str | None,
) -> tuple[Context, bool]:
    """Build the OTel context that places new observations inside ``trace_id``.

    Returns the context plus whether the caller must claim trace root. Langfuse
    fabricates a random parent span id when no real parent is supplied, so the
    observation is a child of something that will never be exported; the public
    SDK path compensates by marking the span as root and this path must do the
    same.
    """
    remote_parent: Final = client._create_remote_parent_span(  # pyright: ignore[reportPrivateUsage]  # no public equivalent in v4
        trace_id=trace_id, parent_span_id=parent_observation_id
    )
    return otel_trace.set_span_in_context(remote_parent), parent_observation_id is None


def start_generation(
    *,
    client: Langfuse,
    context: Context,
    name: str,
    start_time: datetime | None,
    claim_trace_root: bool,
    release: str | None = None,
    attributes: Mapping[str, object],
) -> LangfuseGeneration:
    """Create a generation whose start time is when the model call began.

    No public v4 API accepts a historical start time, so this drives the SDK's
    own OTel tracer, which does. Langfuse documents this route for backdated
    ingestion.
    """
    otel_span: Final = client._otel_tracer.start_span(  # pyright: ignore[reportPrivateUsage]  # only route to a historical start time
        name=name, context=context, start_time=to_unix_nanos(start_time)
    )
    if claim_trace_root:
        otel_span.set_attribute(AS_ROOT_ATTRIBUTE, True)
    generation: Final = LangfuseGeneration(otel_span=otel_span, langfuse_client=client, **attributes)
    if release is not None:
        # after the wrapper, which stamps the client-wide release and would otherwise
        # overwrite the release this request asked for
        otel_span.set_attribute(RELEASE_ATTRIBUTE, release)
    return generation


def start_child_span(
    *,
    client: Langfuse,
    context: Context,
    name: str,
    start_time: datetime | None,
    attributes: Mapping[str, object],
) -> LangfuseSpan:
    """Create a sibling observation inside the same trace, keeping its own window."""
    otel_span: Final = client._otel_tracer.start_span(  # pyright: ignore[reportPrivateUsage]  # only route to a historical start time
        name=name, context=context, start_time=to_unix_nanos(start_time)
    )
    return LangfuseSpan(otel_span=otel_span, langfuse_client=client, **attributes)
