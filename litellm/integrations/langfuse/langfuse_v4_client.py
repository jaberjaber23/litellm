from __future__ import annotations

from typing import Final

import opentelemetry.trace as otel_trace
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

__all__ = [
    "build_isolated_tracer_provider",
    "evict_stale_langfuse_resources",
    "shutdown_langfuse_client",
]

_ENVIRONMENT_ATTRIBUTE: Final = "langfuse.environment"
_RELEASE_ATTRIBUTE: Final = "langfuse.release"


def build_isolated_tracer_provider(*, environment: str | None, release: str | None) -> TracerProvider:
    """Give the langfuse client a provider of its own instead of the process-wide one.

    v4 is built on OpenTelemetry and otherwise either claims the global tracer
    provider, which silently disables litellm's own exporters, or attaches its
    processor to litellm's, which sends litellm spans to every langfuse project
    and langfuse spans to every other litellm destination.

    The resource is rebuilt here because langfuse only applies ``environment``
    and ``release`` when it constructs the provider itself.
    """
    attributes: Final = {
        key: value
        for key, value in ((_ENVIRONMENT_ATTRIBUTE, environment), (_RELEASE_ATTRIBUTE, release))
        if value is not None
    }
    return TracerProvider(resource=Resource.create(attributes))


def evict_stale_langfuse_resources(*, public_key: str | None, secret_key: str | None, base_url: str | None) -> None:
    """Drop a cached client whose credentials no longer match the ones being requested.

    langfuse keys its client registry on the public key alone, so a rotated
    secret or a moved host silently keeps exporting with the original values.
    Only the one stale entry is removed; the SDK's own reset would shut down
    every other tenant in the process.
    """
    if not public_key:
        return
    with LangfuseResourceManager._lock:  # registry has no public accessor
        cached: Final = LangfuseResourceManager._instances.get(public_key)
        if cached is None:
            return
        if getattr(cached, "secret_key", None) == secret_key and getattr(cached, "base_url", None) == base_url:
            return
        LangfuseResourceManager._instances.pop(public_key, None)


def shutdown_langfuse_client(client: Langfuse) -> None:
    """Release everything the client owns, which the SDK's own shutdown does not.

    ``Langfuse.shutdown`` joins the score and media consumers but leaves the
    tracer provider's export thread running and leaves the client in the
    registry, so a later request for the same key gets a dead client back.
    """
    resources: Final = getattr(client, "_resources", None)
    client.flush()
    client.shutdown()
    if resources is None:
        return
    provider: Final = getattr(resources, "tracer_provider", None)
    if provider is not None and not isinstance(provider, otel_trace.ProxyTracerProvider):
        provider.shutdown()
    public_key: Final = getattr(resources, "public_key", None)
    if public_key is None:
        return
    with LangfuseResourceManager._lock:  # registry has no public accessor
        if LangfuseResourceManager._instances.get(public_key) is resources:
            LangfuseResourceManager._instances.pop(public_key, None)
