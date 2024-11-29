# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import getpass
import os
from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from typing import Dict, List

from .constants import (
    DEFAULT_IMAGE_PREFIX,
    DEFAULT_IMAGE_TAG,
    DEFAULT_REGISTRY_PATH,
    FARMVIBES_AI_LOG_LEVEL,
    MAX_WORKER_NODES,
)
from .help_descriptions import (
    ADD_ONNX_HELP,
    ADD_SECRET_HELP,
    DELETE_SECRET_HELP,
    DESTROY_HELP,
    RESTART_HELP,
    SETUP_HELP,
    START_HELP,
    STATUS_HELP,
    STOP_HELP,
    UPDATE_HELP,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 31108
AZURERM_ENVIRONMENTS = [
    "public",
    "usgovernment",
    "german",
    "china",
]

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)
LOCAL_OTEL_PATH = os.path.join(CORE_DIR, "terraform", "local", "modules", "kubernetes", "otel.tf")
REMOTE_OTEL_PATH = os.path.join(CORE_DIR, "terraform", "aks", "modules", "kubernetes", "otel.tf")


class CliParser(ABC):
    SUPPORTED_COMMANDS = [
        ("setup", SETUP_HELP, ["create", "new"]),
        ("update", UPDATE_HELP, ["upgrade", "up"]),
        ("destroy", DESTROY_HELP, ["delete", "remove", "rm"]),
        ("start", START_HELP, ["run"]),
        ("stop", STOP_HELP, ["down", "halt"]),
        ("restart", RESTART_HELP, ["reboot", "reload"]),
        ("status", STATUS_HELP, ["info", "show", "url", "show-url"]),
        ("add-secret", ADD_SECRET_HELP, ["add_secret"]),
        ("delete-secret", DELETE_SECRET_HELP, ["delete_secret"]),
        ("add-onnx", ADD_ONNX_HELP, ["add_onnx", "add-model"]),
    ]

    def __init__(self, name: str):
        self.name = name
        self.commands: Dict[str, argparse.ArgumentParser] = {}
        self.parser = self.build_parser()

    @abstractmethod
    def _add_common_flags(self):
        pass

    def build_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="action", help="Action to perform", required=True)
        for command, help_text, aliases in self.SUPPORTED_COMMANDS:
            self.commands[command] = subparsers.add_parser(command, help=help_text, aliases=aliases)
            self.commands[command].add_argument(
                "--auto-confirm",
                "-y",
                required=False,
                action="store_true",
                help="Answer yes to all prompts",
            )

        self._add_common_flags()
        self._add_flags()
        return parser

    @abstractmethod
    def _add_setup_update_flags(self):
        pass

    def _add_flags(self):
        self._add_setup_update_flags()
        # destroy doesn't need any flags
        # start doesn't need any flags
        # stop doesn't need any flags
        # restart doesn't need any flags
        # status doesn't need any flags
        for secret in (self.commands["add-secret"], self.commands["delete-secret"]):
            secret.add_argument(
                "secret_name",
                help="Name of the secret to operate on",
            )
        self.commands["add-secret"].add_argument(
            "secret_value",
            help="Value of the secret to add",
        )
        self.commands["add-onnx"].add_argument("model_path", help="Path to ONNX model to add")

    def parse(self, args: List[str]):
        return self.parser.parse_args(args)


class LocalCliParser(CliParser):
    def build_parser(self):
        parser = super().build_parser()
        parser.description = (
            "FarmVibes.AI local cluster deployment tool. "
            "Manages a Project FarmVibes.AI cluster on a local machine "
            "by using Docker and Kubernetes."
        )
        return parser

    def _add_flags(self):
        super()._add_flags()
        self.commands["add-onnx"].add_argument(
            "--storage-path",
            type=str,
            default="",
            help="Path to store data needed for cluster operation and output files",
        )

    def _add_setup_update_flags(self):
        for command in (self.commands["setup"], self.commands["update"]):
            command.add_argument(
                "--servers", type=int, default=1, help="Number of servers to create"
            )
            command.add_argument("--agents", type=int, default=0, help="Number of agents to create")
            command.add_argument(
                "--storage-path",
                type=str,
                default="",
                help="Path to store data needed for cluster operation and output files",
            )
            command.add_argument(
                "--registry",
                type=str,
                default=DEFAULT_REGISTRY_PATH.split("/")[0],
                help="Registry to use for images",
            )
            command.add_argument(
                "--registry-username", type=str, default="", help="Username for registry"
            )
            command.add_argument(
                "--registry-password", type=str, default="", help="Password for registry"
            )
            command.add_argument(
                "--image-tag", type=str, default=DEFAULT_IMAGE_TAG, help="Image tag to use"
            )
            command.add_argument(
                "--image-prefix",
                type=str,
                default=DEFAULT_IMAGE_PREFIX,
                help="Prefix to use for images",
            )
            command.add_argument(
                "--log-level",
                type=str,
                default=FARMVIBES_AI_LOG_LEVEL,
                help="Log level to use for FarmVibes.AI services",
            )
            command.add_argument(
                "--max-log-file-bytes",
                type=int,
                default=None,
                help="Maximum size of a log file in bytes.",
            )
            command.add_argument(
                "--log-backup-count",
                type=int,
                default=None,
                help="Number of log files to keep for each service instance.",
            )
            command.add_argument(
                "--worker-replicas",
                type=int,
                default=max(1, cpu_count() // 2 - 1),
                help="Number of worker replicas to use",
            )
            command.add_argument(
                "--port",
                type=int,
                default=DEFAULT_PORT,
                help="Port to use for FarmVibes.AI REST API on host",
            )
            command.add_argument(
                "--host",
                type=str,
                default=DEFAULT_HOST,
                help="Host to use for FarmVibes.AI REST API to bind to",
            )
            command.add_argument(
                "--registry-port",
                type=int,
                default=5000,
                help="Port to use for registry on host",
            )

            if os.path.exists(LOCAL_OTEL_PATH):
                command.add_argument(
                    "--enable-telemetry",
                    default=False,
                    action="store_true",
                    help="Enable telemetry for FarmVibes.AI",
                )

    def _add_common_flags(self):
        cluster_name = os.environ.get(
            "FARMVIBES_AI_CLUSTER_NAME",
            f"farmvibes-ai-{getpass.getuser().replace('_', '-')}",
        )
        for command in self.commands.values():
            command.add_argument(
                "--cluster-name",
                "-n",
                required=False,
                default=cluster_name,
                help="Name of the cluster to operate on",
            )

    def _verify_cluster_name(self, cluster_name: str):
        if "_" in cluster_name:
            raise ValueError(
                f"Invalid character '_' in cluster name '{cluster_name}'. Please, provide a "
                "valid cluster name with the --cluster-name flag or the FARMVIBES_AI_CLUSTER_NAME "
                "environment variable."
            )

    def parse(self, args: List[str]):
        parsed_args = super().parse(args)
        self._verify_cluster_name(parsed_args.cluster_name)
        return parsed_args


class RemoteCliParser(CliParser):
    def build_parser(self):
        parser = super().build_parser()
        parser.description = (
            "FarmVibes.AI remote cluster deployment tool. "
            "Manages a Project FarmVibes.AI cluster on Azure Kubernetes Service."
        )
        return parser

    def _add_common_flags(self):
        for command in self.commands.values():
            command.add_argument(
                "-g",
                "--resource-group",
                required=False,
                help="Azure RG to use (RG will be created if it doesn't exist)",
                default="farmvibes-aks-rg",
            )
            command.add_argument(
                "--cluster-name",
                required=False,
                type=str,
                default="farmvibes-aks",
                help="Name of the cluster to create",
            )
            command.add_argument(
                "-e",
                "--environment",
                required=False,
                choices=AZURERM_ENVIRONMENTS,
                default=AZURERM_ENVIRONMENTS[0],
                help="Azure environment to use",
            )

    def _add_setup_update_flags(self):
        for command in (self.commands["setup"], self.commands["update"]):
            command.add_argument(
                "--cluster-admin-name",
                required=False,
                default="",
                help="Azure username of the cluster admin (overrides automatic detection)",
            )
            command.add_argument(
                "--registry",
                required=False,
                default=DEFAULT_REGISTRY_PATH,
                help="Registry to overwrite where to pull images from",
            )
            command.add_argument(
                "--registry-username", required=False, help="Username for the registry", default=""
            )
            command.add_argument(
                "--registry-password", required=False, help="Password for the registry", default=""
            )
            command.add_argument(
                "--image-prefix",
                required=False,
                help="Prefix for the image names in the registry",
                default=DEFAULT_IMAGE_PREFIX,
            )
            command.add_argument(
                "--image-tag",
                required=False,
                help="Image tags for the images in the registry",
                default=DEFAULT_IMAGE_TAG,
            )
            command.add_argument(
                "--cert-email", required=True, help="Email for the certificate issuing authority"
            )
            command.add_argument("-r", "--region", required=True, help="Azure region")
            command.add_argument(
                "--log-level",
                required=False,
                help="Log level",
                choices=["info", "debug", "warning", "error"],
                default="info",
            )
            command.add_argument(
                "--max-worker-nodes",
                required=False,
                help="Maximum number of (VMs) that support worker nodes",
                default=MAX_WORKER_NODES,
                type=int,
            )
            command.add_argument(
                "--worker-replicas",
                type=int,
                default=3,
                help="Number of worker replicas to use",
            )

            if os.path.exists(REMOTE_OTEL_PATH):
                command.add_argument(
                    "--enable-telemetry",
                    default=False,
                    action="store_true",
                    help="Enable telemetry for FarmVibes.AI",
                )
