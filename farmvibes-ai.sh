#!/bin/bash
# Copyright (c) Microsoft Corporation.

CLIDIR="$(dirname "$0")"/cli

source "${CLIDIR}"/vars.sh

source "${CLIDIR}"/help-functions.sh
source "${CLIDIR}"/helper-functions.sh
source "${CLIDIR}"/dependencies-functions.sh

source "${CLIDIR}"/cluster-functions.sh

source "${CLIDIR}"/cli-functions.sh

## main()
##
##   The entry point for this script.
##
main() {
  if [ "$#" -eq 0 ]; then
    usage
    exit 0
  fi

  if [ $EUID -eq 0 ]; then
    die "This script should not be executed as the root user. Please run it as a regular user. " \
      "If this is being executed with \"sudo\" because of the docker requirement, you should " \
      "probably add your user account to the docker group. For more information, please see " \
      "https://docs.docker.com/engine/install/linux-postinstall/"
  fi

  maybe_process_help "$@"

  mkdir -p "${FARMVIBES_AI_STORAGE_PATH}"
  check_required_tools
  patch_curl
  minikube_search_and_destroy 
  route_command "$@"
}

main "$@"
