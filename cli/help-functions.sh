#!/bin/bash
# Copyright (c) Microsoft Corporation.

## help_exists() command
##
##   Checks whether the help documentation for the command `command` exists.
##
help_exists() {
  local sc=${1:?"Internal error, help_exists() requires a command name"}
  test -e "${ROOTDIR}/documentation/cli/$sc"
  return $?
}

## subcommand_help() command
##
##   Prints the help documentation for the command `command`.
##
subcommand_help() {
  local sc=${1:?"Internal error, help_exists() requires a command name"}
  help_exists "${sc}" || die "Unsupported command ${sc}" && usage
  sed "s/@SCRIPTNAME@/$(basename "${SCRIPTFILE}")/g" < "${ROOTDIR}/documentation/cli/${sc}"
}

## maybe_process_help()
##
##   If the user requested help, prints the help message and exits.
##
maybe_process_help() {
  local help_regex='\s+-h\>'

  if [ "$1" = "-h" ]; then
    # global help, do we have a command?
    shift
    if [ "$1" = "" ]; then
      subcommand_help index
    else
      subcommand_help "${1}"
    fi
    exit 0
  else
    if [[ "$*" =~ $help_regex ]]; then
      subcommand_help "${1}"
      exit 0
    fi
  fi
}

## usage()
##
##   Prints the usage message.
##
usage() {
  echo "usage: $(basename "${SCRIPTFILE}") [-h] start|status|stop|destroy|update"
  echo "       $(basename "${SCRIPTFILE}") [-h] add-secret <key> <value>"
  echo "       $(basename "${SCRIPTFILE}") [-h] delete-secret <key>"
  echo "       $(basename "${SCRIPTFILE}") [-h] setup"
  echo "       $(basename "${SCRIPTFILE}") [-h] add-onnx <model-file>"
  echo
}
