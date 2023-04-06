#!/bin/bash
# Copyright (c) Microsoft Corporation.

## minikube_search_and_destroy()
##
##   Destroys the minikube farmvibes.ai cluster.
##
minikube_search_and_destroy() {
  read -r msg << EOF
There is an old FarmVibes.AI cluster running on minikube.\
To continue using the service, you need to destroy it and create a new K3D cluster.\
This will impact any running workflows. Do you want to proceed?
EOF

  if [[ $(${MINIKUBE} profile list 2> /dev/null | grep "${FARMVIBES_AI_CLUSTER_NAME}") ]]; then
    confirm_action "${msg}" || exit 0

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

  ${K3D} registry list | grep -q "${FARMVIBES_AI_REGISTRY_NAME}.localhost" 2> /dev/null ||
    ${K3D} registry create ${FARMVIBES_AI_REGISTRY_NAME}.localhost --port ${FARMVIBES_AI_REGISTRY_PORT} > /dev/null || \
      die "Failed to create registry. Is something else listening on port ${FARMVIBES_AI_REGISTRY_PORT}?"

  create_data_and_log_dirs

  K3D_FIX_DNS=1 ${K3D} cluster create "${name}" \
    --volume "${FARMVIBES_AI_STORAGE_PATH}:/mnt" \
    --agents 0 \
    --k3s-node-label "agentpool=k3d-${FARMVIBES_AI_CLUSTER_NAME}-server-0@server:0" \
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
}

## restart_services()
##
##   Restarts the services we deployed
restart_services() {
  ${KUBECTL} rollout restart deployment -l backend=terravibes || \
    die "Failed to restart deployments. Is your system under heavy load?"
  ${KUBECTL} rollout status deployment -l backend=terravibes || \
    die "Failed to wait for deployments to restart. Is your system under heavy load?"
}

## deploy_services() initialize
##
##   Deploys all farmvibes.ai services into the currently-logged in cluster.
##   If 'initialize' is set to 1, then the terraform is initialized as 
##   a new cluster. Else just an update of the delta is processed
##
deploy_services() {
  local initialize="${1:?"Internal error, deploy_services() requires if initialization is required"}"

  replicas=$(( $(get_physical_cpus) - 1 ))
  if [[ ${replicas} -lt 1 ]]; then
    echo "WARNING: You have less than 2 CPUs. Setting worker replicas to 1. " \
      "You may face performance issues."
    replicas=1
  fi

  if [[ ${initialize} -eq 1 ]]; then
    ${TERRAFORM} -chdir=${ROOTDIR}/resources/terraform/local init -upgrade
  fi

  ${TERRAFORM} -chdir=${ROOTDIR}/resources/terraform/local apply \
    -state="${FARMVIBES_AI_CONFIG_DIR}/local.tfstate" \
    -auto-approve \
    -var acr_registry="${FARMVIBES_AI_FULL_REGISTRY}" \
    -var run_as_user_id="$(id -u)" \
    -var run_as_group_id="$(id -g)" \
    -var host_assets_dir="${FARMVIBES_AI_DATA_PATH}/assets" \
    -var kubernetes_config_context="k3d-${FARMVIBES_AI_CLUSTER_NAME}" \
    -var image_tag="${FARMVIBES_AI_IMAGE_TAG}" \
    -var node_pool_name="k3d-${FARMVIBES_AI_CLUSTER_NAME}-server-0" \
    -var host_storage_path="/mnt" \
    -var worker_replicas="${replicas}" \
    -var image_prefix="${IMAGES_PREFIX}" \
    -var redis_image_tag="${REDIS_IMAGE_TAG}" \
    -var rabbitmq_image_tag="${RABBITMQ_IMAGE_TAG}" \
    -var farmvibes_log_level="${FARMVIBES_AI_LOG_LEVEL}"
}

## wait_for_deployments()
##
##   Waits for deployments to scale/deploy/update
##
wait_for_deployments() {
  [ -z ${WAIT_AT_THE_END} ] || return
  for fields in "${FARMVIBES_AI_SERVICES[@]}"
  do
    IFS=$'|' read -r deployment tf <<< "$fields"
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
  rm -fr "${ROOTDIR}/resources/terraform/local/.terraform"
  rm -f "${FARMVIBES_AI_CONFIG_DIR}/local.tfstate"
}
