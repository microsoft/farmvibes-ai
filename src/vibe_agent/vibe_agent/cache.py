# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import logging
import os
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Optional, cast

from cloudevents.sdk.event import v1
from dapr.conf import settings
from dapr.ext.grpc import App, TopicEventResponse
from hydra_zen import builds
from opentelemetry import trace

from vibe_common.constants import CACHE_PUBSUB_TOPIC, CONTROL_STATUS_PUBSUB, STATUS_PUBSUB_TOPIC
from vibe_common.dapr import dapr_ready
from vibe_common.messaging import (
    ExecuteRequestContent,
    ExecuteRequestMessage,
    WorkMessage,
    WorkMessageBuilder,
    accept_or_fail_event,
    event_to_work_message,
    extract_message_header_from_event,
    send,
)
from vibe_common.schemas import CacheInfo, OperationSpec, OpRunId
from vibe_common.telemetry import (
    add_span_attributes,
    add_trace,
    get_current_trace_parent,
    setup_telemetry,
    update_telemetry_context,
)
from vibe_core.data.core_types import OpIOType
from vibe_core.logconfig import LOG_BACKUP_COUNT, MAX_LOG_FILE_BYTES, configure_logging

from .cache_metadata_store_client import CacheMetadataStoreClient
from .ops import OperationDependencyResolver
from .ops_helper import OpIOConverter
from .storage.storage import Storage, StorageConfig
from .worker import WorkerMessenger


def get_cache_info(
    dependency_resolver: OperationDependencyResolver,
    input_items: OpIOType,
    op_config: OperationSpec,
    traceparent: str,
) -> CacheInfo:
    # We need traceparent here as abstract event loop mess up the opentelemetry context
    update_telemetry_context(traceparent)

    with trace.get_tracer(__name__).start_as_current_span("get_cache_info"):
        dependencies = dependency_resolver.resolve(op_config)
        stac = OpIOConverter.deserialize_input(input_items)
        cache_info = CacheInfo(op_config.name, op_config.version, stac, dependencies)
        return cache_info


class Cache:
    pubsubname: str
    pre_control_topic: str
    otel_service_name: str

    def __init__(
        self,
        storage: Storage,
        port: int = settings.GRPC_APP_PORT,
        pubsubname: str = CONTROL_STATUS_PUBSUB,
        cache_topic: str = CACHE_PUBSUB_TOPIC,
        status_topic: str = STATUS_PUBSUB_TOPIC,
        logdir: Optional[str] = None,
        max_log_file_bytes: int = MAX_LOG_FILE_BYTES,
        log_backup_count: int = LOG_BACKUP_COUNT,
        loglevel: Optional[str] = None,
        otel_service_name: str = "",
        running_on_azure: bool = False,
    ):
        self.storage = storage
        self.pubsubname = pubsubname
        self.cache_topic = cache_topic
        self.port = port
        self.dependency_resolver = OperationDependencyResolver()
        self.messenger = WorkerMessenger(pubsubname, status_topic)
        self.metadata_store = CacheMetadataStoreClient()
        self.logdir = logdir
        self.loglevel = loglevel
        self.otel_service_name = otel_service_name
        self.max_log_file_bytes = max_log_file_bytes
        self.log_backup_count = log_backup_count
        self.executor = ThreadPoolExecutor() if running_on_azure else ProcessPoolExecutor()
        self.running_on_azure = running_on_azure
        logging.debug(f"Running on azure? {self.running_on_azure}")
        logging.debug(f"Pool type: {type(self.executor)}")

    def retrieve_possible_output(
        self, cache_info: CacheInfo, exec: Executor, traceparent: str
    ) -> Optional[OpIOType]:
        possible_output = self.storage.retrieve_output_from_input_if_exists(cache_info)
        # We need traceparent here as abstract event loop mess up the opentelemetry context
        update_telemetry_context(traceparent)

        with trace.get_tracer(__name__).start_as_current_span("retrieve_possible_output"):
            if possible_output:
                logging.info(f"Cache hit with hash {cache_info.hash} in op {cache_info.name}")
                return OpIOConverter.serialize_output(possible_output)
            logging.info(f"Cache miss with hash {cache_info.hash} in op {cache_info.name}")
            return None

    @add_trace
    def run_new_op(self, message: WorkMessage):
        content = cast(ExecuteRequestContent, message.content)
        add_span_attributes({"op_name": str(content.operation_spec.name)})
        send(
            message,
            self.__class__.__name__.lower(),
            self.pubsubname,
            content.operation_spec.image_name,
        )

        msg = (
            f"Sending new operation to worker. "
            f"Op: {content.operation_spec.name}, "
            f"Params: {content.operation_spec.parameters}, "
            f"Input: {content.operation_spec.inputs_spec}"
        )

        logging.info(msg)

    def fetch_work(self, event: v1.Event) -> TopicEventResponse:
        @add_trace
        def success_callback(message: WorkMessage) -> TopicEventResponse:
            add_span_attributes({"run_id": str(message.header.run_id)})
            content = cast(ExecuteRequestContent, message.content)
            op_config = cast(OperationSpec, content.operation_spec)
            recursion_msg = f"Recursion error for op {op_config.name} - restarting pod."
            try:
                try:
                    cache_info = get_cache_info(
                        self.dependency_resolver,
                        content.input,
                        op_config,
                        get_current_trace_parent(),
                    )
                except RecursionError as e:
                    logging.error(f"{recursion_msg} {e}")
                    os._exit(1)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to get cache info for op {op_config.name} with exception "
                        f"{type(e)}:{e}"
                    ) from e
                possible_output = self.retrieve_possible_output(
                    cache_info, self.executor, get_current_trace_parent()
                )

                async def async_closure():
                    if possible_output is not None:
                        await self.metadata_store.add_refs(
                            str(message.run_id),
                            OpRunId(name=cache_info.name, hash=cache_info.hash),
                            possible_output,
                        )
                        logging.info(f"Cache hit for op {op_config.name}")
                        await self.messenger.send_ack_reply(message)
                        await self.messenger.send_success_reply(
                            message, possible_output, cache_info
                        )
                    else:
                        self.run_new_op(
                            WorkMessageBuilder.add_cache_info_to_execute_request(
                                cast(ExecuteRequestMessage, message), cache_info
                            )
                        )

                asyncio.run(async_closure())
            except RecursionError as e:
                logging.error(f"{recursion_msg} {e}")
                os._exit(1)

            logging.debug(f"Removing message for run_id {message.header.run_id} from queue")
            return TopicEventResponse("success")

        @add_trace
        def failure_callback(event: v1.Event, e: Exception, tb: List[str]) -> TopicEventResponse:
            message = event_to_work_message(event)
            content = cast(ExecuteRequestContent, message.content)
            op_config = cast(OperationSpec, content.operation_spec)
            log_text = f"Failure callback for op {op_config.name}, Exception {e}, Traceback {tb}"
            logging.info(log_text)
            # Send failure reply to orchestrator so we don't get our workflow stuck
            asyncio.run(self.messenger.send_failure_reply(event.id, e, tb))
            return TopicEventResponse("drop")

        update_telemetry_context(extract_message_header_from_event(event).current_trace_parent)

        with trace.get_tracer(__name__).start_as_current_span("fetch_work"):
            return accept_or_fail_event(event, success_callback, failure_callback)  # type: ignore

    def run(self):
        self.app = App()

        appname = f"terravibes-{self.__class__.__name__.lower()}"
        configure_logging(
            default_level=self.loglevel,
            appname=appname,
            logdir=self.logdir,
            max_log_file_bytes=self.max_log_file_bytes,
            log_backup_count=self.log_backup_count,
            logfile=f"{appname}.log",
        )

        if self.otel_service_name:
            setup_telemetry(appname, self.otel_service_name)

        @self.app.subscribe(self.pubsubname, self.cache_topic)
        def fetch_work(event: v1.Event) -> TopicEventResponse:
            return self.fetch_work(event)

        self.start_service()

    @dapr_ready
    def start_service(self):
        logging.info(f"Starting cache listening on port {self.port}")
        self.app.run(self.port)


CacheConfig = builds(
    Cache,
    storage=StorageConfig,
    port=settings.GRPC_APP_PORT,
    pubsubname=CONTROL_STATUS_PUBSUB,
    cache_topic=CACHE_PUBSUB_TOPIC,
    status_topic=STATUS_PUBSUB_TOPIC,
    logdir=None,
    max_log_file_bytes=MAX_LOG_FILE_BYTES,
    log_backup_count=LOG_BACKUP_COUNT,
    loglevel=None,
    otel_service_name="",
    running_on_azure=False,
)
