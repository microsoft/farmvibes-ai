#!/bin/bash
# Copyright (c) Microsoft Corporation.

## check_required_tools()
##
##   Checks whether all tools that we can't install are available in the
##   system.
##
check_required_tools() {
  local tool url
  for fields in "${REQUIRED_TOOLS[@]}"
  do
    IFS=$'|' read -r tool url <<< "$fields"
    if ! command -v "${tool}" > /dev/null; then
      echo "Missing ${tool}. Please see ${url} for instructions on how to install it."
    fi
  done
  if ! docker info > /dev/null 2> /dev/null; then
    die "Unable to talk to docker daemon. Is it running?" \
      "Please make sure you have docker installed and running, and that you have access to it." \
      "For more information, see https://docs.docker.com/get-docker/"
  fi
}

## check_internal_commands()
##
##   Checks whether all internal commands are available in the system.
##
check_internal_commands() {
  local command
  for command in "${INTERNAL_COMMANDS[@]}"
  do
    if ! command -v "${command}" > /dev/null; then
      die Missing "${command}". Please re-run \""$(basename "${SCRIPTFILE}") setup"\" to install it.
    fi
  done
}

## die() [msg]
##
##   Outputs the optional message msg before exiting the script
##
die() {
  msg="${@:-"Something is terribly wrong. Aborting."}"
  echo -e "\033[31;1m${msg} ❌\033[0m"
  exit 1
}

## determine_os()
##
##   Uses `uname` to generate an os name string used across go projects. Exits
##   the script in case it finds an unsupported architecture.
##
determine_os() {
  case $(uname -o) in
      "GNU/Linux")
          echo "linux"
          ;;
      "Darwin")
          echo "darwin"
          ;;
      *)
          die "Unsupported OS. Exiting."
          ;;
  esac
  return 0
}

## determine_arch()
##
##   Uses `uname` to generate an arch string used across go programs. Exits the
##   script in case it finds an unsupported architecture.
##
determine_arch() {
  case $(uname -m) in
      x86_64)
          echo "amd64"
          ;;
      *)
          die "Unsupported architecture. Exiting."
          ;;
  esac
  return 0
}

## get_physical_cpus()
##
##   Determines the number of physical CPUs in the current machine.
##
get_physical_cpus() {
  local cpus
  if command -v lscpu > /dev/null; then
    cpus=$(lscpu -p | grep -Ev '^#' | sort -u -t, -k 2,4 | wc -l)
  else
    cpus=$(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}')
  fi
  number_re='^[0-9]+$'
  if ! [[ ${cpus} =~ ${number_re} ]]; then
    echo "Unable to determine the number physical processors. Using 1." > /dev/stderr
    cpus="1"
  fi
  echo "$cpus"
}

## get_logical_cpus()
##
##   Determines the number of logical CPUs (considering SMT level) in the
##   current machine.
##
get_logical_cpus() {
  local cpus
  if command -v lscpu > /dev/null; then
    cpus=$(lscpu -p | grep -cEv '^#')
  else
    cpus=$(grep -cE ^processor\\s+: /proc/cpuinfo)
  fi
  number_re='^[0-9]+$'
  if ! [[ ${cpus} =~ ${number_re} ]]; then
    echo "Unable to determine the number of CPUs. Using 2." > /dev/stderr
    cpus="2"
  fi
  echo "$cpus"
}

## get_safe_memory_limit()
##
##   Determines a safe amount of memory to allocate to the k3d cluster.
##
get_safe_memory_limit() {
  local mem
  local minimum_mem=$(( 7 * 1024 * 1024 ))
  mem="$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') - 1024 * 1024  - 1))"
  if [[ "$mem" -lt "$minimum_mem" ]]; then
    die "Your host doesn't have enough memory to run farmvibes.ai, stopping installation."
  else
    echo "$mem"
  fi
}

## check_path_sanity() path1 [path2 [path3 ... ]]
##
##   Iterates through a list of paths checking whether they exist and whether
##   we can write to them, aborting if it fails.
##
check_path_sanity() {
  for path in "$@"
  do
    if [ ! -d "${path}" ]; then
      mkdir -p "${path}" || die "Unable to create path ${path}. Aborting installation."
    fi
  done
}

## persist_storage_path()
##
##   Persists the configuration file with the path to the storage used by the
##   cluster.
##
persist_storage_path() {
  echo "${FARMVIBES_AI_STORAGE_PATH}" > "${FARMVIBES_AI_CONFIG_DIR}/${FARMVIBES_AI_DATA_FILE_PATH}"
}

## confirm_action() prompt
##
##   Prompts the user for a yes/no question, returning success (0) on
##   acceptance, and failure on rejection.
##
confirm_action() {
  local yn prompt
  prompt="${@:?"Internal error. confirm_action requires a prompt."}"

  while true; do
    read -rp "$prompt [Y/n] " yn
    yn=${yn,,}  # to lower case
    if [[ "$yn" =~ ^(y| ) ]] || [[ -z "$yn" ]]; then
      return 0
    elif [[ "$yn" =~ ^(n) ]]; then
      return 1
    else
      echo "Please answer with a y/n."
    fi
  done
}