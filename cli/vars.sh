#!/bin/bash
# Copyright (c) Microsoft Corporation.

SCRIPTFILE="$(readlink -f "$0")"
SCRIPTPATH="$(dirname "$SCRIPTFILE")"
ROOTDIR="$(realpath "${SCRIPTPATH}")"

export YAML_PATH="${ROOTDIR}/resources/local-k8s"
export DAPR_YAML_PATH="${YAML_PATH}/dapr-components"

export readonly REQUIRED_TOOLS=(
  'docker|https://docs.docker.com/get-docker/'
  'curl|https://curl.se/ (`apt install curl` in Ubuntu)'
)

export readonly WORKER_YAML="worker.yaml"
export readonly REST_API_YAML="rest-api.yaml"

export readonly DAPR_YAMLS=(
  'rest-orchestrator-pubsub.yaml'
  'statestore.yaml'
  'config.yaml'
  'resiliency.yaml'
  'lockstore.yaml'
)

export readonly CURL_EXTRA_ARGS="--retry 3"

# URLs {{{
export K3D_URL="https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh"
export KUBECTL_BASE_URL="https://storage.googleapis.com/kubernetes-release/release"
export DAPR_URL="https://raw.githubusercontent.com/dapr/cli/master/install/install.sh"
# }}}

export DAPR_RUNTIME_VERSION=1.9.4
export DAPR_DASHBOARD_VERSION=0.10.0
export DAPR_STATEFULSET_DEPENDENCIES=('rabbitmq' 'redis-master')

export K3D_VERSION=v5.4.6
export MINIKUBE_VERSION=v1.26.1
export REDIS_IMAGE_TAG=7.0.4-debian-11-r11
export REDIS_VOL_POD_YAML="${YAML_PATH}/redis-vol-pod.yaml"
export RABBITMQ_IMAGE_TAG=3.10.8-debian-11-r4
export RABBITMQ_SECRET=rabbitmq-connection-string
export RABBITMQ_MAX_TIMEOUT_MS=10800000
export FARMVIBES_AI_CONFIG_DIR="${XDG_CONFIG_HOME:-"${HOME}/.config"}/farmvibes-ai"
export FARMVIBES_AI_DATA_FILE_PATH="storage"
export PATH="${FARMVIBES_AI_CONFIG_DIR}:${PATH}"

if [ -f "${FARMVIBES_AI_CONFIG_DIR}/${FARMVIBES_AI_DATA_FILE_PATH}" ]; then
  export FARMVIBES_AI_STORAGE_PATH=$(cat "${FARMVIBES_AI_CONFIG_DIR}/${FARMVIBES_AI_DATA_FILE_PATH}")
else
  export FARMVIBES_AI_STORAGE_PATH="${FARMVIBES_AI_STORAGE_PATH:-"${HOME}/.cache/farmvibes-ai"}"
  echo "Using ${FARMVIBES_AI_STORAGE_PATH} as storage path." 2> /dev/null
fi

export FARMVIBES_AI_DATA_PATH="${FARMVIBES_AI_STORAGE_PATH}/data"
export FARMVIBES_AI_REDIS_BACKUP_FILE="${FARMVIBES_AI_DATA_PATH}/redis_dump.rdb"
export FARMVIBES_AI_CLUSTER_NAME=farmvibes-ai
export FARMVIBES_AI_REST_API_NAME=terravibes-rest-api
export FARMVIBES_AI_ONNX_RESOURCES="${FARMVIBES_AI_STORAGE_PATH}/onnx_resources"

# Internal commands {{{
export DAPR="${FARMVIBES_AI_CONFIG_DIR}/dapr"
export HELM="${FARMVIBES_AI_CONFIG_DIR}/helm"
export K3D="${FARMVIBES_AI_CONFIG_DIR}/k3d"
export MINIKUBE="${FARMVIBES_AI_CONFIG_DIR}/minikube"
export KUBECTL="${FARMVIBES_AI_CONFIG_DIR}/kubectl"
export CURL="${FARMVIBES_AI_CONFIG_DIR}/curl"

export INTERNAL_COMMANDS=("${DAPR}" "${HELM}" "${K3D}" "${KUBECTL}" "${CURL}")
# }}}

export FARMVIBES_AI_REGISTRY_NAME="${FARMVIBES_AI_CLUSTER_NAME}-registry"
export FARMVIBES_AI_REGISTRY_PORT="5000"
export FARMVIBES_AI_FULL_REGISTRY_NAME="k3d-${FARMVIBES_AI_REGISTRY_NAME}.localhost"
export FARMVIBES_AI_FULL_REGISTRY="${FARMVIBES_AI_FULL_REGISTRY_NAME}:${FARMVIBES_AI_REGISTRY_PORT}"

if [ -z "${CLIDIR}" ]; then
  . cli/cr-vars.sh
else
  . "${CLIDIR}/cr-vars.sh"
fi
