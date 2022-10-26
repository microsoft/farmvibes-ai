#!/bin/bash
# Copyright (c) Microsoft Corporation.

SCRIPTFILE="$(readlink -f "$0")"
SCRIPTPATH="$(dirname "$SCRIPTFILE")"
ROOTDIR="$(realpath "${SCRIPTPATH}")"

export YAML_PATH="${ROOTDIR}/resources/minikube"
export DAPR_YAML_PATH="${YAML_PATH}/dapr-components"

export readonly REQUIRED_TOOLS=(
  'docker|https://docs.docker.com/get-docker/'
  'curl|https://curl.se/ (`apt install curl` in Ubuntu)'
)

export readonly WORKER_YAML="worker.yaml"
export readonly REST_API_YAML="rest-api.yaml"
export readonly FARMVIBES_AI_YAMLS='rest-api.yaml|orchestrator.yaml|worker.yaml|cache.yaml'

export readonly DAPR_YAMLS=('rest-orchestrator-pubsub.yaml' 'statestore.yaml')

# URLs {{{
export MINIKUBE_BASE_URL="https://storage.googleapis.com/minikube/releases"
export KUBECTL_BASE_URL="https://storage.googleapis.com/kubernetes-release/release"
export DAPR_URL="https://raw.githubusercontent.com/dapr/cli/master/install/install.sh"
# }}}

export DAPR_RUNTIME_VERSION=1.8.4
export DAPR_DASHBOARD_VERSION=0.10.0

export MINIKUBE_VERSION=v1.26.1
export REDIS_IMAGE_TAG=7.0.4-debian-11-r11
export RABBITMQ_IMAGE_TAG=3.10.8-debian-11-r4
export RABBITMQ_SECRET=rabbitmq-connection-string
export FARMVIBES_AI_CONFIG_DIR="${XDG_CONFIG_HOME:-"${HOME}/.config"}/farmvibes-ai"
export FARMVIBES_AI_DATA_FILE_PATH="storage"

if [ -f "${FARMVIBES_AI_CONFIG_DIR}/${FARMVIBES_AI_DATA_FILE_PATH}" ]; then
  export FARMVIBES_AI_STORAGE_PATH=$(cat "${FARMVIBES_AI_CONFIG_DIR}/${FARMVIBES_AI_DATA_FILE_PATH}")
else
  export FARMVIBES_AI_STORAGE_PATH="${FARMVIBES_AI_STORAGE_PATH:-"${HOME}/.cache/farmvibes-ai"}"
  echo "Using ${FARMVIBES_AI_STORAGE_PATH} as storage path." 2> /dev/null
fi

export FARMVIBES_AI_DATA_PATH="${FARMVIBES_AI_STORAGE_PATH}/data"
export FARMVIBES_AI_CLUSTER_NAME=farmvibes-ai
export FARMVIBES_AI_REST_API_NAME=farmvibes-ai-rest-api
export FARMVIBES_AI_DEPLOYMENTS=('terravibes-rest-api' 'terravibes-orchestrator' 'terravibes-worker' 'terravibes-cache')
export FARMVIBES_AI_ONNX_RESOURCES="${FARMVIBES_AI_STORAGE_PATH}/onnx_resources"

# Internal commands {{{
export DAPR="${FARMVIBES_AI_CONFIG_DIR}/dapr"
export HELM="${FARMVIBES_AI_CONFIG_DIR}/helm"
export MINIKUBE="${FARMVIBES_AI_CONFIG_DIR}/minikube"
export KUBECTL="${FARMVIBES_AI_CONFIG_DIR}/kubectl"

export INTERNAL_COMMANDS=("${DAPR}" "${HELM}" "${MINIKUBE}" "${KUBECTL}")
# }}}

if [ -z "${CLIDIR}" ]; then
  . cli/cr-vars.sh
else
  . "${CLIDIR}/cr-vars.sh"
fi
