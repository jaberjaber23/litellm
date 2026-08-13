from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import opentelemetry.trace as otel_trace
from langfuse import LangfuseGeneration, LangfuseSpan

if TYPE_CHECKING:
    from langfuse import Langfuse

    Span = Any
else:
    Span = Any

AS_ROOT_ATTRIBUTE: Final = "langfuse.internal.as_root"


def to_unix_nanos(value: datetime | None) -> int | None:
    """Langfuse v4 takes OTel timestamps, which are integer nanoseconds since the epoch."""
    if value is None:
        return None
    return int(value.timestamp() * 1_000_000_000)


def open_trace_context(
    *,
    client: Langfuse,
    trace_id: str,
    parent_observation_id: str | None,
) -> tuple[Any, bool]:
    """Build the OTel context that places new observations inside ``trace_id``.

    Returns the context plus whether the caller must claim trace root. Langfuse
    fabricates a random parent span id when no real parent is supplied, so the
    observation is a child of something that will never be exported; the public
    SDK path compensates by marking the span as root and this path must do the
    same.
    """
    remote_parent: Final = client._create_remote_parent_span(  # no public equivalent in v4
        trace_id=trace_id, parent_span_id=parent_observation_id
    )
    return otel_trace.set_span_in_context(remote_parent), parent_observation_id is None


def start_generation(
    *,
    client: Langfuse,
    context: Any,
    name: str,
    start_time: datetime | None,
    claim_trace_root: bool,
    attributes: dict[str, Any],
) -> LangfuseGeneration:
    """Create a generation whose start time is when the model call began.

    No public v4 API accepts a historical start time, so this drives the SDK's
    own OTel tracer, which does. Langfuse documents this route for backdated
    ingestion.
    """
    otel_span: Final = client._otel_tracer.start_span(  # only route to a historical start time
        name=name, context=context, start_time=to_unix_nanos(start_time)
    )
    if claim_trace_root:
        otel_span.set_attribute(AS_ROOT_ATTRIBUTE, True)
    return LangfuseGeneration(otel_span=otel_span, langfuse_client=client, **attributes)


def start_child_span(
    *,
    client: Langfuse,
    context: Any,
    name: str,
    start_time: datetime | None,
    attributes: dict[str, Any],
) -> LangfuseSpan:
    """Create a sibling observation inside the same trace, keeping its own window."""
    otel_span: Final = client._otel_tracer.start_span(  # only route to a historical start time
        name=name, context=context, start_time=to_unix_nanos(start_time)
    )
    return LangfuseSpan(otel_span=otel_span, langfuse_client=client, **attributes)
