# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

DEFAULT_IMAGE_PREFIX = "farmai/terravibes/"
DEFAULT_IMAGE_TAG = "12088305617"
DEFAULT_REGISTRY_PATH = "mcr.microsoft.com"

LOCAL_SERVICE_URL_PATH_FILE = "service_url"
REMOTE_SERVICE_URL_PATH_FILE = "remote_service_url"
MAX_WORKER_NODES = 3

AZURE_CR_DOMAIN = "azurecr.io"

# Local constants
ONNX_SUBDIR = "onnx_resources"
FARMVIBES_AI_LOG_LEVEL = "DEBUG"
REDIS_IMAGE = (
    "docker.io/library/redis:7.4.10-bookworm"
    "@sha256:e9b2e45ecd47fbb69b877cf8d045d5cccaaaed52524b6e098b4abe8212994f73"
)
RABBITMQ_IMAGE = (
    "docker.io/library/rabbitmq:4.3.5-management"
    "@sha256:397fde82bc04522d88680b57cbf5d70caae715a76c957404e52e3f0fa056b8f3"
)
