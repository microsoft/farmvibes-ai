#!/bin/bash
# Copyright (c) Microsoft Corporation.

# Container registry variables
export CONTAINER_REGISTRY_BASE="${CONTAINER_REGISTRY_BASE:-"mcr.microsoft.com/farmai/terravibes"}"
export IMAGES="api-orchestrator worker"
export IMAGES_PREFIX="${FARMVIBES_AI_IMAGE_PREFIX:-""}"
export FARMVIBES_AI_IMAGE_TAG=${FARMVIBES_AI_IMAGE_TAG:-"prod"}
