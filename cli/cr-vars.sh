#!/bin/bash
# Copyright (c) Microsoft Corporation.

# Container registry variables
export CONTAINER_REGISTRY_BASE="${CONTAINER_REGISTRY_BASE:-"mcr.microsoft.com/farmai/terravibes"}"
export IMAGES="api-orchestrator worker cache"
export IMAGES_PREFIX="${FARMVIBES_AI_IMAGE_PREFIX:-""}"
export FARMVIBES_AI_IMAGE_TAG=${FARMVIBES_AI_IMAGE_TAG:-"2023-02-16"}
export readonly FARMVIBES_AI_DEPLOYMENTS=(
  'terravibes-rest-api|rest-api.yaml'
  'terravibes-orchestrator|orchestrator.yaml'
  'terravibes-cache|cache.yaml'
  'terravibes-worker|worker.yaml'
)
