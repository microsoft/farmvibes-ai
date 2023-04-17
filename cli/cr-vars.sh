#!/bin/bash
# Copyright (c) Microsoft Corporation.

# Container registry variables
export CONTAINER_REGISTRY_BASE="${CONTAINER_REGISTRY_BASE:-"mcr.microsoft.com/farmai/terravibes"}"
export IMAGES="api-orchestrator worker cache"
export IMAGES_PREFIX="${FARMVIBES_AI_IMAGE_PREFIX:-""}"
export FARMVIBES_AI_IMAGE_TAG=${FARMVIBES_AI_IMAGE_TAG:-"2023.04.17"}
export readonly FARMVIBES_AI_SERVICES=(
  'terravibes-rest-api|restapi.tf'
  'terravibes-orchestrator|orchestrator.tf'
  'terravibes-cache|cache.tf'
  'terravibes-worker|worker.tf'
)
export FARMVIBES_AI_LOG_LEVEL=${FARMVIBES_AI_LOG_LEVEL:-"INFO"}
