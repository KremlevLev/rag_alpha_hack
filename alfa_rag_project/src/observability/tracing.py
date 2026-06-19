"""
Observability for the RAG pipeline.

Supports Arize Phoenix (OpenTelemetry OTLP), LangSmith, and a no-op fallback.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class Tracer(ABC):
    """Abstract tracer for pipeline observability."""

    @abstractmethod
    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        ...

    @abstractmethod
    def end_trace(self, trace_id: str, status: str = "ok") -> None:
        ...

    @abstractmethod
    def log_event(self, trace_id: str, event: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        ...

    @contextmanager
    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        trace_id = self.start_trace(name, metadata)
        try:
            yield trace_id
        except Exception as exc:
            self.end_trace(trace_id, status="error")
            self.log_event(trace_id, "error", {"message": str(exc)})
            raise
        else:
            self.end_trace(trace_id, status="ok")


class NullTracer(Tracer):
    """No-op tracer for local development without observability infrastructure."""

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return uuid.uuid4().hex[:16]

    def end_trace(self, trace_id: str, status: str = "ok") -> None:
        pass

    def log_event(self, trace_id: str, event: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class PhoenixTracer(Tracer):
    """
    Arize Phoenix tracer using OpenTelemetry.

    Sends spans to a local Phoenix collector via OTLP gRPC.
    Requires: opentelemetry-exporter-otlp-proto-grpc
    """

    def __init__(self, endpoint: str = "http://localhost:4317", service_name: str = "enterprise-rag") -> None:
        self._endpoint = endpoint
        self._service_name = service_name
        self._tracer = None
        self._spans: dict[str, Any] = {}

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        tracer = self._get_tracer()
        span = tracer.start_span(name)
        trace_id = uuid.uuid4().hex[:16]
        self._spans[trace_id] = span
        if metadata:
            for k, v in metadata.items():
                span.set_attribute(k, str(v))
        return trace_id

    def end_trace(self, trace_id: str, status: str = "ok") -> None:
        span = self._spans.pop(trace_id, None)
        if span:
            span.set_attribute("status", status)
            span.end()

    def log_event(self, trace_id: str, event: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        span = self._spans.get(trace_id)
        if span:
            attrs = attributes or {}
            span.add_event(event, attrs)

    def _get_tracer(self):
        if self._tracer is None:
            try:
                from opentelemetry import trace
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                resource = Resource.create({"service.name": self._service_name})
                provider = TracerProvider(resource=resource)
                exporter = OTLPSpanExporter(endpoint=self._endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                trace.set_tracer_provider(provider)
                self._tracer = trace.get_tracer(__name__)
            except ImportError as exc:
                logger.warning("OpenTelemetry not installed, falling back to NullTracer: %s", exc)
                return NullTracer()
        return self._tracer


class LangSmithTracer(Tracer):
    """
    LangSmith tracer.

    Requires: langsmith, LANGSMITH_API_KEY env var.
    """

    def __init__(self, api_key: Optional[str] = None, project: Optional[str] = None) -> None:
        self._api_key = api_key
        self._project = project
        self._client = None
        self._runs: dict[str, Any] = {}

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        client = self._get_client()
        if client is None:
            return uuid.uuid4().hex[:16]

        from langsmith import run_helpers

        run_id = uuid.uuid4().hex[:16]
        run = run_helpers.create_run(
            client=client,
            name=name,
            run_type="chain",
            inputs=metadata or {},
            project_name=self._project,
        )
        self._runs[run_id] = run
        return run_id

    def end_trace(self, trace_id: str, status: str = "ok") -> None:
        run = self._runs.pop(trace_id, None)
        if run:
            from langsmith import run_helpers

            run_helpers.update_run(
                client=self._get_client(),
                run_id=run.id,
                outputs={"status": status},
            )

    def log_event(self, trace_id: str, event: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass  # LangSmith does not support mid-run events easily

    def _get_client(self):
        if self._client is None and self._api_key:
            try:
                from langsmith import Client as LangSmithClient

                self._client = LangSmithClient(api_key=self._api_key)
            except ImportError:
                logger.warning("langsmith not installed")
        return self._client


# ── Factory ───────────────────────────────────────────────────


def create_tracer(provider: str = "null", **kwargs) -> Tracer:
    if provider == "phoenix":
        return PhoenixTracer(**kwargs)
    elif provider == "langsmith":
        return LangSmithTracer(**kwargs)
    else:
        return NullTracer(**kwargs)