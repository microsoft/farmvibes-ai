#!/bin/bash
# Copyright (c) Microsoft Corporation.

## setup_minikube_cluster() name
##
##   Creates a new minikube cluster with name "name".
##
setup_minikube_cluster() {
  local name="${1:?"Internal error, setup_minikube_cluster() requires a cluster name"}"

  ${MINIKUBE} profile list 2> /dev/null | grep -q "${name}" && return 0

  ${MINIKUBE} start --driver=docker --profile="${name}" \
    --interactive=false --delete-on-failure=true --container-runtime=docker \
    --extra-config=apiserver.service-node-port-range=3000-32767 \
    --mount --mount-uid="$(id -u)" --mount-gid="$(id -g)" \
    --mount-string="${FARMVIBES_AI_STORAGE_PATH}:/mnt" \
    --disable-metrics \
    --memory="$(get_safe_memory_limit)"k --cpus="$(get_logical_cpus)"
}

## show_service_url()
##
##   Outputs the URL of the service.
##
show_service_url() {
  local nodeport ip

  ip="$(docker network inspect "${FARMVIBES_AI_CLUSTER_NAME}" -f "{{range \$i, \$value := .Containers}}{{if eq \$value.Name \"${FARMVIBES_AI_CLUSTER_NAME}\"}}{{println .IPv4Address}}{{end}}{{end}}" | cut -d / -f 1)"
 
  nodeport="$(${KUBECTL} get service terravibes-rest-api -o jsonpath='{.spec.ports[].nodePort}')"
  url="http://${ip}:${nodeport}"

  if [ -z "$ip" ]; then
    url=$(${MINIKUBE} service "${FARMVIBES_AI_REST_API_NAME}" --url --profile="${FARMVIBES_AI_CLUSTER_NAME}")
  fi

  echo "${url}" > "${FARMVIBES_AI_CONFIG_DIR}/service_url" 2> /dev/null
  echo "FarmVibes.AI REST API is running at ${url}"
}

## pull_and_sideload_images() registry images tag prefix
##
##   Pulls images from a container registry
pull_and_sideload_images() {
  local registry="$1"
  local images="$2"
  local tag="$3"
  local prefix="$4"

  for image in $images
  do
    local fullname="${registry}/${prefix}${image}:${tag}"
    docker pull "${fullname}" || \
      die "Failed to pull ${fullname}. Make sure your network is up, that you have disk space," \
        "that you can access the container registry ${registry}, and that the image" \
        "${prefix}${image}:${tag} exists."
  done
}

## build_k8s_cluster()
##
##   Builds a new kubernetes cluster we can use to deploy farmvibes.ai on.
##
build_k8s_cluster() {
  setup_minikube_cluster "${FARMVIBES_AI_CLUSTER_NAME}" || die \
    "Not overwriting existing cluster ${FARMVIBES_AI_CLUSTER_NAME}." \
    "If you want to recreate it, please destroy it first." \
    "(${SCRIPTFILE} destroy)"

  PATH="${FARMVIBES_AI_CONFIG_DIR}:$PATH" ${DAPR} init \
    --runtime-version "${DAPR_RUNTIME_VERSION}" \
    --dashboard-version "${DAPR_DASHBOARD_VERSION}" -k
}

## update_images()
##
##   Updates images in an existing farmvibes.ai kubernetes cluster.
##
update_images() {
  ${MINIKUBE} profile list 2> /dev/null | grep -q "${FARMVIBES_AI_CLUSTER_NAME}" || \
    die "No farmvibes.ai cluster found"

  (\
    eval "$("${MINIKUBE}" --profile="${FARMVIBES_AI_CLUSTER_NAME}" docker-env)" && \
    pull_and_sideload_images "${CONTAINER_REGISTRY_BASE}" "${IMAGES}" "${FARMVIBES_AI_IMAGE_TAG}" "${IMAGES_PREFIX}"
  ) || die "Failed to download Farmvibes.AI images."

  for deployment in $(${KUBECTL} get deployments -l backend=terravibes -o name | cut -d / -f 2 | xargs)
  do
    ${KUBECTL} delete deployment "${deployment}"
  done

  deploy_services
}

## restart_services()
##
##   Restarts the services we deployed
restart_services() {
  local replicas
  for deployment in "${FARMVIBES_AI_DEPLOYMENTS[@]}"
  do
    replicas="$(${KUBECTL} get deployment "${deployment}" -o jsonpath="{.status.replicas}")"
    if [ -z "$replicas" ]; then
      replicas=1
    fi
    ${KUBECTL} scale deployment "${deployment}" --replicas=0
    ${KUBECTL} delete pod -l app="${deployment}" --wait=true --grace-period=1 || \
      die "Failed to update ${deployment} deployment"
    ${KUBECTL} scale deployment "${deployment}" --replicas="${replicas}"
    ${KUBECTL} rollout status deployment "$deployment"
  done
}

## deploy_services()
##
##   Deploys all farmvibes.ai services into the currently-logged in cluster.
##
deploy_services() {
  install_redis
  install_rabbitmq

  local contents image replicas
  for dapr_component in "${DAPR_YAMLS[@]}"
  do
    ${KUBECTL} apply -f "${DAPR_YAML_PATH}/${dapr_component}"
  done

  echo "${FARMVIBES_AI_YAMLS}" | tr '|' '\n' | while read -r yaml
  do
    if [[ "$yaml" == "$WORKER_YAML" ]]; then
      replicas=$(( $(get_physical_cpus) - 1 ))
      if [[ "$replicas" -lt 1 ]]; then
        die "Not enough processors for running FarmVibes.AI. Cancelling installation."
      fi
      contents=$(sed "s/replicas: REPLICAS_TO_BE_REPLACED/replicas: ${replicas}/g" < "${YAML_PATH}/${yaml}")
      contents=$(echo "${contents}" | USER_ID=$(id -u) GROUP_ID=$(id -g) envsubst)
    elif [[ "$yaml" == "$REST_API_YAML" ]]; then
      contents=$(sed "s|FARMVIBES_AI_HOST_ASSETS_DIR|${FARMVIBES_AI_DATA_PATH}/assets|g" < "${YAML_PATH}/${yaml}")
    else
      contents=$(<"${YAML_PATH}/${yaml}")
    fi
    image=$(grep -E 'image:.*:latest' <<< "${contents}" | rev | cut -d / -f 1 | rev | cut -d : -f 1)
    sed "s|\\(image:\\s\\+\\).*|\\1${CONTAINER_REGISTRY_BASE}/${IMAGES_PREFIX}${image}:${FARMVIBES_AI_IMAGE_TAG}|" <<< "${contents}" | \
      ${KUBECTL} apply -f -
  done
}

## wait_for_deployments()
##
##   Waits for deployments to scale/deploy/update
##
wait_for_deployments() {
  for deployment in "${FARMVIBES_AI_DEPLOYMENTS[@]}"
  do
    ${KUBECTL} wait --for=condition=Available deployment --timeout=90s "$deployment"
    ${KUBECTL} rollout status deployment "$deployment"
  done
}

## get_cluster_status()
##
##   Gets the current status of the cluster
##
get_cluster_status() {
  status=$(${MINIKUBE} profile list 2> /dev/null | \
    grep -E "(${FARMVIBES_AI_CLUSTER_NAME}|Profile)" -B 1 -A 1 \
    || die "No farmvibes.ai cluster found"
  )
  echo "${status}"
}

## is_cluster_running()
##
##   Checks whether a cluster is running.
##
is_cluster_running() {
  local status running_re

  running_re='[rR]unning'
  status=$(get_cluster_status)
  if [[ "${status}" =~ $running_re ]]; then
    return 0
  fi
  return 1
}

## stop_cluster()
##
##   Stops a cluster.
##
stop_cluster() {
  ${MINIKUBE} stop --profile="${FARMVIBES_AI_CLUSTER_NAME}" | grep -v minikube || \
    die "Failed to stop farmvibes.ai cluster"
}

## destroy_cluster()
##
##   Destroys a cluster.
##
destroy_cluster() {
  docker rm "${FARMVIBES_AI_CLUSTER_NAME}" &> /dev/null
  ${MINIKUBE} delete --profile="${FARMVIBES_AI_CLUSTER_NAME}" | grep -v minikube
}
