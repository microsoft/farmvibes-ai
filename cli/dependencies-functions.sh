#!/bin/bash
# Copyright (c) Microsoft Corporation.


## build_minikube_binary_name()
##
##   Builds the binary name of the minikube release according to the minikube project
##   rules.
##
build_minikube_binary_name() {
  local version=${1:-"${MINIKUBE_VERSION}"}
  local arch=$(determine_arch)
  local os=$(determine_os)

  echo "minikube-${os}-${arch}"
}

## build_minikube_url() [version]
##
##   Builds an URL for downloading MINIKUBE `version`, which defaults to
##   $MINIKUBE_VERSION.
##
build_minikube_url() {
  local version=${1:-"${MINIKUBE_VERSION}"}
  local binary
  binary=$(build_minikube_binary_name)

  echo "${MINIKUBE_BASE_URL}/${version}/${binary}"
}

## install_minikube() minikube_binary_name [path]
##
##   Installs the minikube binary built for the OS os in architecture arch to
##   optional path. If the path is not provided, the script uses
##   $FARMVIBES_AI_CONFIG_DIR/minikube.
##
install_minikube() {
  local binary=${1:?"Internal error, install_minikube() requires a binary name"}
  local path=${2:-"${FARMVIBES_AI_CONFIG_DIR}/minikube"}

  local base url
  base=$(dirname "${path}")
  url=$(build_minikube_url)

  if [ -f "${path}" ]; then
    return 0
  fi

  echo "Installing minikube at ${path}..."
  if [ ! -d "${base}" ]; then
    mkdir -p "${base}" || die "Failed to create local bin path ${base}"
    mkdir -p "${FARMVIBES_AI_STORAGE_PATH}" || die "Failed to create local bin path ${FARMVIBES_AI_STORAGE_PATH}"
  fi
  curl -L "${url}" -o "${base}/${binary}" 2> /dev/null
  chmod +x "${base}/${binary}"
  mv "${base}/${binary}" "${path}"
}

## install_kubectl() [path]
##
## Installs the kubectl binary built for the current OS and arch to optional
## path. If the path is not provided, the script uses $FARMVIBES_AI_CONFIG_DIR/kubectl.
##
install_kubectl() {
  local path=${1:-"${FARMVIBES_AI_CONFIG_DIR}/kubectl"}
  local latest_release=$(curl -s ${KUBECTL_BASE_URL}/stable.txt)

  local os=$(determine_os)
  local arch=$(determine_arch)
  local full_url="${KUBECTL_BASE_URL}/${latest_release}/bin/${os}/${arch}/kubectl"

  if [ ! -f "${path}" ]; then
    echo "Installing kubectl at ${path}..."
    curl -L "${full_url}" -o ${path}
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
    curl -Ls "${releases_url}" | \
    grep 'href="/helm/helm/releases/tag/v3.[0-9]*.[0-9]*\"' | \
    sed -E 's/.*\/helm\/helm\/releases\/tag\/(v[0-9\.]+)".*/\1/g' | \
    head -1 \
  )

  local helm_dist="helm-${latest_release}-${os}-${arch}.tar.gz"
  local helm_url="https://get.helm.sh/${helm_dist}"
  local tempdir="$(mktemp -d)"
  local out="${tempdir}/helm.tar.gz"

  curl -sSL "${helm_url}" -o "${out}" 2> /dev/null
  tar xf "${out}" -C "${tempdir}"

  mv "${tempdir}/${os}-${arch}/helm" "${path}"
  rm -fr "${tempdir}"
}

## install_redis()
##
##   Uses the bitnami helm chart to install redis in the current k8s cluster.
##
install_redis() {
  echo "Installing redis in the cluster..."
  ${HELM} repo add bitnami https://charts.bitnami.com/bitnami > /dev/null || \
    die "Failed to add redis helm chart"
  ${HELM} repo update > /dev/null || \
    die "Failed to update helm repo"
  ${HELM} install redis --set image.tag="${REDIS_IMAGE_TAG}" bitnami/redis > /dev/null || \
    die "Failed to install redis in k8s cluster"
  ${KUBECTL} scale --replicas 0 statefulsets/redis-replicas
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
    curl -L "${DAPR_URL}" 2> /dev/null | DAPR_INSTALL_DIR="$(dirname "${path}")" /bin/bash > /dev/null
  fi
}

## install_dependencies()
##
##   Downloads from the internet all the binary tools we need to create and
##   maintain a local Farmvibes.AI cluster
install_dependencies() {
  mkdir -p "${FARMVIBES_AI_CONFIG_DIR}"
  install_dapr || die "Failed to install dapr. Aborting..."
  install_minikube "$(build_minikube_binary_name)" || die "Failed to install minikube. Aborting..."
  install_kubectl || die "Failed to install kubectl. Aborting..."
  install_helm || die "Failed to install helm. Aborting..."

  echo "FarmVibes.AI dependencies installed!"
}
