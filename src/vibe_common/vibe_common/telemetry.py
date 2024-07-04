import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict

from opentelemetry import trace
from opentelemetry.context import attach
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.span import INVALID_SPAN

from vibe_common.constants import TRACEPARENT_STRING

LOGGER = logging.getLogger(__name__)


def setup_telemetry(service_name: str, exporter_endpoint: str):
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Create an OTLP exporter instance
    # The insecure=True flag is used here because we're running the
    # service locally (from the k8s cluster perspective) without
    # Transport Layer Security (TLS).
    otlp_exporter = OTLPSpanExporter(endpoint=exporter_endpoint, insecure=True)

    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Sets the global default tracer provider
    trace.set_tracer_provider(provider)


def get_current_trace_parent() -> str:
    span = trace.get_current_span()

    if span == INVALID_SPAN:
        LOGGER.warning("No current span found. Returning empty traceparent.")

    trace_id = span.get_span_context().trace_id
    span_id = span.get_span_context().span_id
    trace_flags = span.get_span_context().trace_flags
    return TRACEPARENT_STRING.format(trace_id=trace_id, parent_id=span_id, trace_flags=trace_flags)


def add_span_attributes(attributes: Dict[str, Any]):
    current_span = trace.get_current_span()
    for k, v in attributes.items():
        current_span.set_attribute(k, v)


def update_telemetry_context(trace_parent: str):
    """Updates the current telemetry context with the trace parent"""
    attach(extract({"traceparent": trace_parent}))


def add_trace(func: Callable[..., Any]):
    if inspect.iscoroutinefunction(func):
        return _add_trace_async(func)
    else:
        return _add_trace_sync(func)


def _add_trace_sync(func: Callable[..., Any]):
    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def _add_trace_async(func: Callable[..., Any]):
    @wraps(func)
    async def wrapper(*args, **kwargs):  # type: ignore
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(func.__name__):
            return await func(*args, **kwargs)

    return wrapper
