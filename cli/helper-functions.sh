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
      die "Missing ${tool}. Please see ${url} for instructions on how to install it."
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
  local command installer
  for command in "${INTERNAL_COMMANDS[@]}"
  do
    if ! command -v "${command}" > /dev/null; then
      installer="install_$(basename ${command})"
      [[ $(type -t "$installer") == "function" ]] && $installer || die \
        Missing "${command}". Please re-run \""$(basename "${SCRIPTFILE}") setup"\" to install it.
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
  if [[ "$cpus" -gt "$MAXIMUM_DEFAULT_WORKERS" ]]; then
    cpus="$MAXIMUM_DEFAULT_WORKERS"
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

## confirm_action() prompt [default]
##
##   Prompts the user for a yes/no question, returning success (0) on
##   acceptance, and failure on rejection.
##
confirm_action() {
  local yn prompt default confirm
  prompt="${1:?"Internal error. confirm_action requires a prompt."}"
  default="${2:-y}"
  default="${default^}"  # to upper case

  if [[ $default == "Y" ]]; then
    confirm="[Y/n]"
  else
    confirm="[y/N]"
  fi

  while true; do
    read -p "${prompt} ${confirm} " -r yn
    yn=$(echo "${yn,,}" | xargs)  # to lower case
    if [[ -z "$yn" || "$yn" == " " ]]; then
      yn="${default,,}"
    fi
    if [[ "$yn" =~ ^(y| ) ]]; then
      return 0
    elif [[ "$yn" =~ ^(n) ]]; then
      return 1
    else
      echo "Please answer with a y/n."
    fi
  done
}

## patch_curl
##
##   Creates a new `curl` executable in the config dir with additional curl
##   arguments we'd like to use.
##
patch_curl() {
  if [ -f "${CURL}" ]; then
    return
  fi

  if [ ! -d "${FARMVIBES_AI_CONFIG_DIR}" ]; then
    mkdir -p "${FARMVIBES_AI_CONFIG_DIR}"
  fi

  cat << EOF > "${CURL}"
#!/bin/sh

exec $(which curl) ${CURL_EXTRA_ARGS} \$@ 
EOF

  chmod +x "${CURL}"
}

## get_pip_install_command
##
##   Determines whether we're running in a conda venv, a python venv, or not,
##   and returns the pip command to run to install the client.
##   If no venv is detected we use `pip install --user`. Otherwise, we use `pip
##   install
##
get_pip_install_command() {
  if [[ ! -z $CONDA_DEFAULT_ENV || ! -z $VIRTUAL_ENV ]]; then
    echo "pip install"
    return
  fi
  echo "pip install --user"
}

## update_or_not
##
##   Determines whether we should use `--upgrade` with `pip install`
##   for upgrading the vibe_core library.
##
upgrade_or_not() {
  python -c "import vibe_core" 2> /dev/null && echo "--upgrade"
}

## install_or_update_client
##
##   Installs or updates the vibe-core library. Assumes we are running from
##   a git repo.
##
install_or_update_client() {
  $(get_pip_install_command) $(upgrade_or_not) $ROOTDIR/src/vibe_core 2> /dev/null
}


## check_docker_free_space()
##
##   Checks whether the partition that holds the docker root has at 
##   least 30G of free space, or 5%.
##
check_docker_free_space() {
  local docker_root docker_root_partition free_space min_free_space min_free_space_percentage free_space_percentage

  if [ ! -z "${FARMVIBES_AI_SKIP_DOCKER_FREE_SPACE_CHECK}" ]; then
    return
  fi

  docker_root=$(docker info -f '{{.DockerRootDir}}')
  docker_root_partition=$(df -P "${docker_root}" | tail -1 | awk '{print $1}')
  free_space=$(df -P "${docker_root_partition}" | tail -1 | awk '{print $4}')
  min_free_space=30000000
  max_use_percentage=95
  used_space_percentage=$(df -P "${docker_root_partition}" | tail -1 | awk '{print $5}' | sed 's/%//')

  if [ "${free_space}" -lt "${min_free_space}" ] || [ "${used_space_percentage}" -gt "${max_use_percentage}" ]; then
    echo "WARNING: The partition that holds the docker root has less than 30GB (or 5%) of free space." | fold -s
    echo "This may cause the cluster to fail to start."
    echo "You can free some space by removing old docker images and containers."
    echo "You can also change the docker root directory by editing the docker daemon configuration file."
    echo "See https://docs.docker.com/config/daemon/systemd/#runtime-directory-and-storage-driver"
    echo "for more information."
    confirm_action "Do you want to continue?" "n" || exit 1
  fi
}
