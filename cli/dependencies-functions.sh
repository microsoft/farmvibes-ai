#!/bin/bash
# Copyright (c) Microsoft Corporation.

## install_terraform() [path]
##
##   Installs the terraform binary built for the OS os in architecture arch to
##   optional path. If the path is not provided, the script uses
##   $FARMVIBES_AI_CONFIG_DIR/terraform.
##
install_terraform() {
  local path=${1:-"${FARMVIBES_AI_CONFIG_DIR}/terraform.zip"}
  local exe_path=${TERRAFORM}

  local os=$(determine_os)
  local arch=$(determine_arch)
  local full_url="https://releases.hashicorp.com/terraform/1.3.9/terraform_1.3.9_${os}_${arch}.zip"

  if [ ! -f "${path}" ]; then
    echo "Downloading terraform at ${path}..."
    ${CURL} -L "${full_url}" -o ${path}
    unzip -o ${path} -d ${FARMVIBES_AI_CONFIG_DIR}
    rm ${path}
  fi

  chmod +x "${exe_path}"
}

## install_k3d() [path]
##
##   Installs the k3d binary built for the OS os in architecture arch to
##   optional path. If the path is not provided, the script uses
##   $FARMVIBES_AI_CONFIG_DIR/k3d.
##
install_k3d() {
  local path=${1:-"${FARMVIBES_AI_CONFIG_DIR}/k3d"}

  local base
  base=$(dirname "${path}")

  if [ -f "${path}" ]; then
    return 0
  fi

  echo "Installing k3d at ${path}..."
  if [ ! -d "${base}" ]; then
    mkdir -p "${base}" || die "Failed to create local bin path ${base}"
    mkdir -p "${FARMVIBES_AI_STORAGE_PATH}" || die "Failed to create local bin path ${FARMVIBES_AI_STORAGE_PATH}"
  fi

  ${CURL} -sL "${K3D_URL}" | env USE_SUDO="false" TAG="$K3D_VERSION" K3D_INSTALL_DIR="$base" bash
}

## install_kubectl() [path]
##
## Installs the kubectl binary built for the current OS and arch to optional
## path. If the path is not provided, the script uses $FARMVIBES_AI_CONFIG_DIR/kubectl.
##
install_kubectl() {
  local path=${1:-"${FARMVIBES_AI_CONFIG_DIR}/kubectl"}
  local latest_release=$(${CURL} -s ${KUBECTL_BASE_URL}/stable.txt)

  local os=$(determine_os)
  local arch=$(determine_arch)
  local full_url="${KUBECTL_BASE_URL}/${latest_release}/bin/${os}/${arch}/kubectl"

  if [ ! -f "${path}" ]; then
    echo "Installing kubectl at ${path}..."
    ${CURL} -L "${full_url}" -o ${path}
  fi

  chmod +x "${path}"
}

## install_helm() [path]
##
## Installs the helm binary built for the current OS and arch to optional path.
## If the path is not provided, the script uses $FARMVIBES_AI_CONFIG_DIR/kubectl.
##
install_helm() {
  local path=${1:-"${FARMVIBES_AI_CONFIG_DIR}/helm"}
  if [ -f "${path}" ]; then
    return 0
  fi
  echo "Installing helm at ${path}..."

  local os=$(determine_os)
  local arch=$(determine_arch)

  local releases_url="https://github.com/helm/helm/releases"
  local latest_release=$(\
    ${CURL} -Ls "${releases_url}" | \
    grep 'href="/helm/helm/releases/tag/v3.[0-9]*.[0-9]*\"' | \
    sed -E 's/.*\/helm\/helm\/releases\/tag\/(v[0-9\.]+)".*/\1/g' | \
    head -1 \
  )

  local helm_dist="helm-${latest_release}-${os}-${arch}.tar.gz"
  local helm_url="https://get.helm.sh/${helm_dist}"
  local tempdir="$(mktemp -d)"
  local out="${tempdir}/helm.tar.gz"

  ${CURL} -sSL "${helm_url}" -o "${out}" 2> /dev/null
  tar xf "${out}" -C "${tempdir}"

  mv "${tempdir}/${os}-${arch}/helm" "${path}"
  rm -fr "${tempdir}"
}


## has_stateful_set [ß]
##
##   Returns 0 when stateful set ß is present in the cluster
##
has_stateful_set() {
  ${KUBECTL} get statefulset "$1" > /dev/null 2> /dev/null && return 0 || return 1
}

## backup_redis_data()
##
##   Store redis data before destroying the cluster.
##
backup_redis_data() {
  local pod_name redis_password

  # Read the redis master pod name
  pod_name=$(${KUBECTL} get pods --no-headers -o custom-columns=":metadata.name" -l app.kubernetes.io/component=master)

  # Read the redis password
  redis_password=$(${KUBECTL} get secret redis -o jsonpath="{.data.redis-password}" | base64 --decode)

  # Set the append only configuration to false
  ${KUBECTL} exec ${pod_name} -- bash -c "echo -e 'AUTH ${redis_password}\nCONFIG SET appendonly no\nsave' | redis-cli"

  # Save redis data on the host machine
  ${KUBECTL} cp ${pod_name}:/data/dump.rdb ${FARMVIBES_AI_REDIS_BACKUP_FILE} -c redis && \
    echo "Saved redis data to ${FARMVIBES_AI_REDIS_BACKUP_FILE}"
}

## restore_redis_data()
##
##   Restore redis data after creating a new cluster
##
restore_redis_data() {

  local pod_name redis_password

  # Just ask for redis data restoration if there is a
  # previous dump.
  if [[ $(ls ${FARMVIBES_AI_REDIS_BACKUP_FILE} 2> /dev/null) ]]; then

    confirm_action "Do you want to restore the workflow execution records from a previous cluster?" || return 0

    # Read the redis master pod name
    pod_name=$(${KUBECTL} get pods --no-headers -o custom-columns=":metadata.name" -l app.kubernetes.io/component=master)

    # Read the redis password
    redis_password=$(${KUBECTL} get secret redis -o jsonpath="{.data.redis-password}" | base64 --decode)

    # Turn the redis-master off
    ${KUBECTL} scale --replicas 0 statefulsets/redis-master

    # Create a dummy pod to copy the saved dump.
    # This is the process recommended by bitnamy docs
    # https://docs.bitnami.com/kubernetes/infrastructure/redis/administration/backup-restore/
    ${KUBECTL} apply -f ${REDIS_VOL_POD_YAML}

    # Wait the dummy pod to be running
    ${KUBECTL} wait --for=jsonpath='{.status.phase}'=Running --timeout=120s pod/redisvolpod

    # Copy the redis dump to the persistent volume
    ${KUBECTL} cp ${FARMVIBES_AI_REDIS_BACKUP_FILE} redisvolpod:/mnt/dump.rdb

    # Delete the dummy pod
    ${KUBECTL} delete pod/redisvolpod

    # Restart redis
    ${KUBECTL} scale --replicas 1 statefulsets/redis-master
  fi
}

## increase_rabbit_timeout()
##
##   Increases consumer timeout in RabbitMQ
##
increase_rabbit_timeout() {
  ${KUBECTL} rollout status statefulset rabbitmq
  ${KUBECTL} exec -i rabbitmq-0 -- rabbitmqctl eval "application:set_env(rabbit, consumer_timeout, ${RABBITMQ_MAX_TIMEOUT_MS})."
}

## install_dapr() [path]
##
##   Downloads the dapr CLI and installs the dapr control plane in the current
##   k8s cluster.
##
install_dapr() {
  local path=${1:-"${FARMVIBES_AI_CONFIG_DIR}/dapr"}

  if ! command -v "${DAPR}" > /dev/null; then
    echo "Installing dapr CLI at ${path}..."
    ${CURL} -L "${DAPR_URL}" 2> /dev/null | DAPR_INSTALL_DIR="$(dirname "${path}")" /bin/bash > /dev/null
  fi
}

## install_keda()
##
##   Deploys keda in the current cluster.
##
install_keda() {
  has_deployment keda-operator keda && return

  ${HELM} repo add kedacore https://kedacore.github.io/charts
  ${HELM} repo update
  ${KUBECTL} create namespace keda
  ${HELM} install keda --set image.tag="${KEDA_VERSION}" kedacore/keda --namespace keda
}

## install_dependencies()
##
##   Downloads from the internet all the binary tools we need to create and
##   maintain a local Farmvibes.AI cluster
install_dependencies() {
  mkdir -p "${FARMVIBES_AI_CONFIG_DIR}"
  install_dapr || die "Failed to install dapr. Aborting..."
  install_k3d || die "Failed to install k3d. Aborting..."
  install_kubectl || die "Failed to install kubectl. Aborting..."
  install_helm || die "Failed to install helm. Aborting..."
  install_terraform || die "Failed to install terraform. Aborting..."

  echo "FarmVibes.AI dependencies installed!"
}

## create_data_and_log_dirs()
##
##   Creates the data directories for the cluster.
##
create_data_and_log_dirs() {
  for dir in "${FARMVIBES_AI_DATA_DIRS[@]}"
  do
    mkdir -p "${FARMVIBES_AI_STORAGE_PATH}/${dir}" || \
      die "Failed to create ${dir} directory. Do you have permissions to write "\
        "in ${FARMVIBES_AI_STORAGE_PATH}?"
  done
  for fields in "${FARMVIBES_AI_DEPLOYMENTS[@]}"
  do
    IFS=$'|' read -r deployment yaml <<< "$fields"
    mkdir -p "${FARMVIBES_AI_STORAGE_PATH}/logs/${deployment}"
  done
}
