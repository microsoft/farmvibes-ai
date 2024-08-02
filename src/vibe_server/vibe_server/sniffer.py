# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
from typing import List

from cloudevents.sdk.event import v1
from dapr.conf import settings
from dapr.ext.grpc import App

from vibe_common.constants import (
    CACHE_PUBSUB_TOPIC,
    CONTROL_PUBSUB_TOPIC,
    CONTROL_STATUS_PUBSUB,
    STATUS_PUBSUB_TOPIC,
)
from vibe_common.dapr import dapr_ready
from vibe_common.messaging import event_to_work_message
from vibe_core.logconfig import LOG_BACKUP_COUNT, MAX_LOG_FILE_BYTES, configure_logging


class Sniffer:
    app: App
    topics: List[str]

    def __init__(self, pubsub: str, topics: List[str], port: int = settings.GRPC_APP_PORT):
        self.app = App()
        self.port = port
        self.pubsub = pubsub
        self.topics = topics
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.logger.info(f"Will subscribe to topics {topics}")
        for topic in self.topics:

            @self.app.subscribe(self.pubsub, topic)
            def log(event: v1.Event) -> None:
                self.log(event, topic)

    def log(self, event: v1.Event, topic: str) -> None:
        try:
            message = event_to_work_message(event)
        except Exception:
            raise RuntimeError(f"Failed to decode event with id {event.id}")
        self.logger.info(f"{event.source} => {topic}: {message}")

    @dapr_ready
    def run(self):
        self.app.run(self.port)


def main():
    parser = argparse.ArgumentParser(
        "vibe-sniffer", description="Sniffs TerraVibes queues and logs them"
    )

    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help=(
            "Path to which to save logs "
            "(if specified, duplicate messages will be persisted for all services)"
        ),
    )
    parser.add_argument(
        "--max-log-file-bytes",
        type=int,
        help="The maximum number of bytes for a log file",
        default=MAX_LOG_FILE_BYTES,
    )
    parser.add_argument(
        "--log-backup-count",
        type=int,
        help="The number of log files to keep",
        required=False,
        default=LOG_BACKUP_COUNT,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to listen on for dapr connections",
    )
    parser.add_argument(
        "--pubsub",
        type=str,
        default=CONTROL_STATUS_PUBSUB,
        help="dapr pubsub to connect to",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=[CONTROL_PUBSUB_TOPIC, CACHE_PUBSUB_TOPIC, STATUS_PUBSUB_TOPIC],
        help="the topics to sniff",
    )
    args = parser.parse_args()

    configure_logging(
        logdir=None if args.logdir is None else args.logdir,
        max_log_file_bytes=args.max_log_file_bytes if args.max_log_file_bytes else None,
        log_backup_count=args.log_backup_count if args.log_backup_count else None,
        appname="sniffer",
    )

    sniffer = Sniffer(
        pubsub=args.pubsub,
        topics=args.topics,
        port=args.port,
    )
    sniffer.run()


if __name__ == "__main__":
    main()
