#!/bin/bash
# Copyright (c) Microsoft Corporation.

## minikube_search_and_destroy()
##
##   Destroys the minikube farmvibes.ai cluster.
##
minikube_search_and_destroy() {
  if [[ $(${MINIKUBE} profile list 2> /dev/null | grep "${FARMVIBES_AI_CLUSTER_NAME}") ]]; then
    confirm_action "There is a FarmVibes.AI minikube cluster running." \
      "To continue using the service, you need to destroy it and create a new K3D cluster."\
      "This could impact any running workflow. Do you want to proceed?" || exit 0
    
    backup_redis_data

    ${MINIKUBE} stop --profile="${FARMVIBES_AI_CLUSTER_NAME}" | grep -v minikube || \
      die "Failed to stop farmvibes.ai cluster"

    docker rm "${FARMVIBES_AI_CLUSTER_NAME}" &> /dev/null
    ${MINIKUBE} delete --profile="${FARMVIBES_AI_CLUSTER_NAME}" | grep -v minikube

    rm -f "${FARMVIBES_AI_CONFIG_DIR}/${FARMVIBES_AI_DATA_FILE_PATH}"

    echo "Minikube cluster has been deleted"
  fi 
}

## setup_k3d_cluster() name
##
##   Creates a new k3d cluster with name "name".
##
setup_k3d_cluster() {
  local name="${1:?"Internal error, setup_k3d_cluster() requires a cluster name"}"

  ${K3D} cluster list 2> /dev/null | grep -q "${name}" && return 0

  k3d registry list | grep -q "${FARMVIBES_AI_REGISTRY_NAME}.localhost" 2> /dev/null ||
    ${K3D} registry create ${FARMVIBES_AI_REGISTRY_NAME}.localhost --port ${FARMVIBES_AI_REGISTRY_PORT} > /dev/null || \
      die "Failed to create registry. Is something else listening on port ${FARMVIBES_AI_REGISTRY_PORT}?"

  ${K3D} cluster create "${name}" \
    --volume "${FARMVIBES_AI_STORAGE_PATH}:/mnt" \
    --agents 0 \
    --registry-use "${FARMVIBES_AI_FULL_REGISTRY}"
}

## show_service_url()
##
##   Outputs the URL of the service.
##
show_service_url() {
  local nodeport ip

  ip="$(docker network inspect "k3d-${FARMVIBES_AI_CLUSTER_NAME}" -f "{{range \$i, \$value := .Containers}}{{if eq \$value.Name \"k3d-${FARMVIBES_AI_CLUSTER_NAME}-server-0\"}}{{println .IPv4Address}}{{end}}{{end}}" | cut -d / -f 1)"
 
  nodeport="$(${KUBECTL} get service terravibes-rest-api -o jsonpath='{.spec.ports[].nodePort}')"
  url="http://${ip}:${nodeport}"

  echo "${url}" > "${FARMVIBES_AI_CONFIG_DIR}/service_url" 2> /dev/null
  if [ "${ip}" ]; then
    echo "FarmVibes.AI REST API is running at ${url}"
  fi
}

## transform_image_name() [fullname]
##
##   Transforms `fullname` to a unique name in our cluster for tagging
##
transform_image_name() {
  local fullname="${1:?"Internal error, transform_image_name() requires a cluster name"}"

  fullname=$(echo "${fullname}" | rev | cut -d / -f 1 | rev)

  echo "${FARMVIBES_AI_FULL_REGISTRY}/${fullname}"
}

## localhost_ip_image_name() fullname
##
##   Transforms `fullname` into a registry reference for pushing to 127.0.0.1.
##
localhost_ip_image_name() {
  local fullname="${1:?"Internal error, transform_image_name() requires a cluster name"}"

  FARMVIBES_AI_FULL_REGISTRY="127.0.0.1:${FARMVIBES_AI_REGISTRY_PORT}" transform_image_name "${fullname}"
}

## pull_and_sideload_images() registry images tag prefix
##
##   Pulls images from a container registry
pull_and_sideload_images() {
  local registry="$1"
  local images="$2"
  local tag="$3"
  local prefix="$4"
  local local_name

  for image in $images
  do
    local fullname="${registry}/${prefix}${image}:${tag}"
    local_name=$(localhost_ip_image_name "${fullname}")
    docker pull "${fullname}" || \
      die "Failed to pull ${fullname}. Make sure your network is up, that you have disk space," \
        "that you can access the container registry ${registry}, and that the image" \
        "${prefix}${image}:${tag} exists."
    docker tag "${fullname}" "${local_name}"
    echo "Pushing downloaded image ${fullname} into our local cluster as ${local_name}"
    docker push "${local_name}"
  done
}

## build_k8s_cluster()
##
##   Builds a new kubernetes cluster we can use to deploy farmvibes.ai on.
##
build_k8s_cluster() {
  setup_k3d_cluster "${FARMVIBES_AI_CLUSTER_NAME}" || die \
    "Not overwriting existing cluster ${FARMVIBES_AI_CLUSTER_NAME}." \
    "If you want to recreate it, please destroy it first." \
    "(${SCRIPTFILE} destroy)"
}

## update_images()
##
##   Updates images in an existing farmvibes.ai kubernetes cluster.
##
update_images() {
  ${K3D} cluster list 2> /dev/null | grep -q "${FARMVIBES_AI_CLUSTER_NAME}" || \
    die "No farmvibes.ai cluster found"

  pull_and_sideload_images "${CONTAINER_REGISTRY_BASE}" "${IMAGES}" "${FARMVIBES_AI_IMAGE_TAG}" "${IMAGES_PREFIX}" \
    || die "Failed to download Farmvibes.AI images."

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
  install_dapr_in_cluster
  install_redis
  install_rabbitmq

  local contents image replicas
  for dapr_component in "${DAPR_YAMLS[@]}"
  do
    ${KUBECTL} apply -f "${DAPR_YAML_PATH}/${dapr_component}"
  done

  replicas=$(( $(get_physical_cpus) - 1 ))
  if [[ "$replicas" -lt 1 ]]; then
    die "Not enough processors for running FarmVibes.AI. Cancelling installation."
  fi

  echo "${FARMVIBES_AI_YAMLS}" | tr '|' '\n' | while read -r yaml
  do
    contents=$(sed "s/replicas: REPLICAS_TO_BE_REPLACED/replicas: ${replicas}/g" < "${YAML_PATH}/${yaml}")
    contents=$(USER_ID=$(id -u) GROUP_ID=$(id -g) envsubst <<< "${contents}")
    if [[ "$yaml" == "$REST_API_YAML" ]]; then
      contents=$(sed "s|FARMVIBES_AI_HOST_ASSETS_DIR|${FARMVIBES_AI_DATA_PATH}/assets|g" <<< "${contents}")
    fi
    image=$(grep -E 'image:.*:latest' <<< "${contents}" | rev | cut -d / -f 1 | rev | cut -d : -f 1)
    fullname="${CONTAINER_REGISTRY_BASE}/${IMAGES_PREFIX}${image}:${FARMVIBES_AI_IMAGE_TAG}"
    local_name=$(transform_image_name "${fullname}")
    sed "s|\\(image:\\s\\+\\).*|\\1${local_name}|" <<< "${contents}" | \
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
  status=$(${K3D} cluster list | grep ${FARMVIBES_AI_CLUSTER_NAME} 2> /dev/null | \
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

  running_re='1/1'
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
  ${K3D} cluster stop "${FARMVIBES_AI_CLUSTER_NAME}" || \
    die "Failed to stop farmvibes.ai cluster"
}

## destroy_cluster()
##
##   Destroys a cluster.
##
destroy_cluster() {
  backup_redis_data
  ${K3D} cluster delete "${FARMVIBES_AI_CLUSTER_NAME}" || \
    die "Failed to delete farmvibes.ai cluster"
  ${K3D} registry delete "${FARMVIBES_AI_FULL_REGISTRY_NAME}"
}
