#!/usr/bin/env bash

SCRIPTFILE=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPTFILE")
ROOTDIR=$(realpath $SCRIPTPATH/..)
DEV_ENV_FILE=$ROOTDIR/resources/envs/dev.yaml

conda env update -f $DEV_ENV_FILE

# Installing internal packages
terravibes_packages="vibe_core vibe_common vibe_agent vibe_server vibe_lib vibe_dev"
for package in $terravibes_packages; do
    echo Installing package $package
    pip install -e $ROOTDIR/src/$package
done