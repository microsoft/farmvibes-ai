.PHONY: help local clean revert-% revert clean-% local-% cluster set-image %-base

SHELL = /bin/bash

export PATH := $(HOME)/.config/farmvibes-ai:$(PATH)

CACHE_DEPLOYMENT := terravibes-cache
ORCHESTRATOR_DEPLOYMENT := terravibes-orchestrator
REST_API_DEPLOYMENT := terravibes-rest-api
DATA_OPS_DEPLOYMENT := terravibes-data-ops
WORKER_DEPLOYMENT := terravibes-worker

CACHE_REPO := farmai/terravibes/cache
ORCHESTRATOR_REPO := farmai/terravibes/api-orchestrator
REST_API_REPO := farmai/terravibes/api-orchestrator
DATA_OPS_REPO := farmai/terravibes/cache
WORKER_REPO := farmai/terravibes/worker

CONTAINER_DEBUG_PORT := 5678
REST_API_DEBUG_PORT := 5678
ORCHESTRATOR_DEBUG_PORT := 5679
CACHE_DEBUG_PORT := 5680
WORKER_DEBUG_PORT := 5681
DATA_OPS_DEBUG_PORT := 5682

CURRENT_CACHE_REPLICAS := $(shell env PATH=$(PATH) kubectl get deployment $(CACHE_DEPLOYMENT) -o jsonpath='{.status.replicas}')
CURRENT_REST_API_REPLICAS := $(shell env PATH=$(PATH) kubectl get deployment $(REST_API_DEPLOYMENT) -o jsonpath='{.status.replicas}')
CURRENT_ORCHESTRATOR_REPLICAS := $(shell env PATH=$(PATH) kubectl get deployment $(ORCHESTRATOR_DEPLOYMENT) -o jsonpath='{.status.replicas}')
CURRENT_DATA_OPS_REPLICAS := $(shell env PATH=$(PATH) kubectl get deployment $(DATA_OPS_DEPLOYMENT) -o jsonpath='{.status.replicas}')
CURRENT_WORKER_REPLICAS := $(shell env PATH=$(PATH) kubectl get deployment $(WORKER_DEPLOYMENT) -o jsonpath='{.status.replicas}')

TAG := tmp-$(shell date +%s)
ROOT := $(shell git rev-parse --show-toplevel)

build_cluster := env FARMVIBES_AI_IMAGE_PREFIX=terravibes- CONTAINER_REGISTRY_BASE=mcr.microsoft.com bash farmvibes-ai local setup
base_image_name := grep -oE 'FROM ([-a-zA-Z0-9@:%._\+~\#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~\#?&//=]*))' FILE | cut -d ' ' -f 2

define transform_image_name
$(shell docker ps | grep registry | rev | cut -d ' ' -f 1 | rev):5000/$(1)
endef

help: ## Shows this help message
	@echo -e This is the farmvibes.ai makefile. Supported targets are:\\n
	@grep -E -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

local: cluster restore-git-lfs local-rest-api local-cache local-worker local-orchestrator local-data-ops ## Builds all images locally and deploys them into the local farmvibes.ai cluster
	[ -z $(WAIT_AT_THE_END) ] || kubectl delete pods -l backend=terravibes && \
		kubectl wait --for=condition=Available deployment --timeout=300s -l backend=terravibes

revert: cluster revert-rest-api revert-cache revert-worker revert-orchestrator ## Reverts all images to the official version

restore-git-lfs:
	git lfs pull

services-base: resources/docker/Dockerfile-services-base
	@docker manifest inspect `$(subst FILE,$<,$(base_image_name))` || \
		az acr login -n `$(subst FILE,$<,$(base_image_name)) | cut -d / -f 1 | sed 's|.azurecr.io||g'` || \
		echo "Failed to log into container registry. Please perform an `az login` and try again"

%-base: resources/docker/Dockerfile-%
	@docker manifest inspect `$(subst FILE,$<,$(base_image_name))` || \
		az acr login -n `$(subst FILE,$<,$(base_image_name)) | cut -d / -f 1 | sed 's|.azurecr.io||g'` || \
		echo "Failed to log into container registry. Please perform an `az login` and try again"

delete-%:
	kubectl scale deployment $(subst delete-,,$@) --replicas=0
	kubectl delete pod --wait=true -l app=$(subst delete-,,$@) --grace-period=0 --force
	kubectl rollout status deployment $(subst delete-,,$@)

repo-%:
	docker pull $(CONTAINER_REGISTRY_BASE)/$(subst repo-,,$@):$(FARMVIBES_AI_IMAGE_TAG)

set-image:
	kubectl set image deployment $(DEPLOYMENT) "*=$(IMAGE_FULL_REFERENCE)"
	kubectl rollout status deployment $(DEPLOYMENT)

set-registry-image: push-image
	DEPLOYMENT=$(DEPLOYMENT) IMAGE_FULL_REFERENCE=$(call transform_image_name,$(IMAGE_FULL_REFERENCE)) make -C . set-image

push-image:
	docker tag $(IMAGE_FULL_REFERENCE) 127.0.0.1:5000/$(IMAGE_FULL_REFERENCE)
	docker push 127.0.0.1:5000/$(IMAGE_FULL_REFERENCE)

scale:
	kubectl scale deployment $(DEPLOYMENT) --replicas=$(shell [ "$(REPLICAS)" ] && echo "$(REPLICAS)" || echo 1)
	[ ! -z $(WAIT_AT_THE_END) ] || kubectl wait --for=condition=Available deployment --timeout=300s $(DEPLOYMENT)

# Have to replace Xfrozen_modules=on with Xfrozen_modules=off in the deployment
disable-frozen-modules:
	kubectl get deployment $(DEPLOYMENT) -o yaml | sed 's|Xfrozen_modules=on|Xfrozen_modules=off|g' | kubectl apply -f -

add-debug-flag:
	kubectl get deployment $(DEPLOYMENT) -o yaml | sed 's|\(\s\+-\)\(.*port=3000\)|\1\2\n\1 --debug|' | kubectl apply -f -

add-debug-flag-agent:
	kubectl get deployment $(DEPLOYMENT) -o yaml | sed 's|\(\s\+-\)\(.*port=3000\)|\1\2\n\1 debug.activate=true|' | kubectl apply -f -

local-rest-api: cluster local-rest-api-orchestrator delete-$(REST_API_DEPLOYMENT) ## Builds and deploys a local REST API image (enabling debug)
	DEPLOYMENT=$(REST_API_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(REST_API_REPO):$(TAG) $(MAKE) -C . set-registry-image
	@kubectl get deployment $(REST_API_DEPLOYMENT) -o json | grep -v last | grep -qo -- --debug || DEPLOYMENT=$(REST_API_DEPLOYMENT) $(MAKE) -C . add-debug-flag
	DEPLOYMENT=$(REST_API_DEPLOYMENT) $(MAKE) -C . disable-frozen-modules
	DEPLOYMENT=$(REST_API_DEPLOYMENT) REPLICAS=$(CURRENT_REST_API_REPLICAS) $(MAKE) scale

revert-rest-api: cluster repo-$(REST_API_REPO) delete-$(REST_API_DEPLOYMENT) ## Reverts the REST API deployment to use the official image
	DEPLOYMENT=$(REST_API_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(CONTAINER_REGISTRY_BASE)/$(REST_API_REPO):$(FARMVIBES_AI_IMAGE_TAG) $(MAKE) set-registry-image
	DEPLOYMENT=$(REST_API_DEPLOYMENT) REPLICAS=$(CURRENT_REST_API_REPLICAS) make scale

local-orchestrator: cluster local-rest-api-orchestrator delete-$(ORCHESTRATOR_DEPLOYMENT) ## Builds and deploys a local ORCHESTRATOR image (enabling debug)
	DEPLOYMENT=$(ORCHESTRATOR_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(ORCHESTRATOR_REPO):$(TAG) $(MAKE) -C . set-registry-image
	@kubectl get deployment $(ORCHESTRATOR_DEPLOYMENT) -o json | grep -v last | grep -qo -- --debug || DEPLOYMENT=$(ORCHESTRATOR_DEPLOYMENT) $(MAKE) -C . add-debug-flag
	DEPLOYMENT=$(ORCHESTRATOR_DEPLOYMENT) $(MAKE) -C . disable-frozen-modules
	DEPLOYMENT=$(ORCHESTRATOR_DEPLOYMENT) REPLICAS=$(CURRENT_ORCHESTRATOR_REPLICAS) $(MAKE) scale

revert-orchestrator: cluster repo-$(ORCHESTRATOR_REPO) delete-$(ORCHESTRATOR_DEPLOYMENT) ## Reverts the ORCHESTRATOR deployment to use the official image
	DEPLOYMENT=$(ORCHESTRATOR_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(CONTAINER_REGISTRY_BASE)/$(ORCHESTRATOR_REPO):$(FARMVIBES_AI_IMAGE_TAG) $(MAKE) set-registry-image
	DEPLOYMENT=$(ORCHESTRATOR_DEPLOYMENT) REPLICAS=$(CURRENT_ORCHESTRATOR_REPLICAS) make scale

local-data-ops: cluster local-cache-repo delete-$(DATA_OPS_DEPLOYMENT) ## Builds and deploys a local data ops image (enabling debug)
	DEPLOYMENT=$(DATA_OPS_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(DATA_OPS_REPO):$(TAG) $(MAKE) -C . set-registry-image
	@kubectl get deployment $(DATA_OPS_DEPLOYMENT) -o json | grep -v last | grep -qo debug.activate || DEPLOYMENT=$(DATA_OPS_DEPLOYMENT) $(MAKE) -C . add-debug-flag-agent
	DEPLOYMENT=$(DATA_OPS_DEPLOYMENT) $(MAKE) -C . disable-frozen-modules
	DEPLOYMENT=$(DATA_OPS_DEPLOYMENT) REPLICAS=$(CURRENT_DATA_OPS_REPLICAS) $(MAKE) scale

revert-data-ops: cluster repo-$(DATA_OPS_REPO) delete-$(DATA_OPS_DEPLOYMENT) ## Reverts the data ops deployment to use the official image
	DEPLOYMENT=$(DATA_OPS_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(CONTAINER_REGISTRY_BASE)/$(DATA_OPS_REPO):$(FARMVIBES_AI_IMAGE_TAG) $(MAKE) set-registry-image
	DEPLOYMENT=$(DATA_OPS_DEPLOYMENT) REPLICAS=$(CURRENT_DATA_OPS_REPLICAS) make scale

local-worker: cluster restore-git-lfs local-worker-repo delete-$(WORKER_DEPLOYMENT) ## Builds and deploys a local WORKER image (enabling debug)
	DEPLOYMENT=$(WORKER_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(WORKER_REPO):$(TAG) $(MAKE) -C . set-registry-image
	DEPLOYMENT=$(WORKER_DEPLOYMENT) $(MAKE) -C . disable-frozen-modules
	DEPLOYMENT=$(WORKER_DEPLOYMENT) REPLICAS=$(CURRENT_WORKER_REPLICAS) make scale

revert-worker: cluster repo-$(WORKER_REPO) delete-$(WORKER_DEPLOYMENT) ## Reverts the WORKER deployment to use the official image
	DEPLOYMENT=$(WORKER_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(CONTAINER_REGISTRY_BASE)/$(WORKER_REPO):$(FARMVIBES_AI_IMAGE_TAG) make set-registry-image
	DEPLOYMENT=$(WORKER_DEPLOYMENT) REPLICAS=$(CURRENT_WORKER_REPLICAS) make scale

local-cache: cluster local-cache-repo delete-$(CACHE_DEPLOYMENT) ## Builds and deploys a local CACHE image (enabling debug)
	DEPLOYMENT=$(CACHE_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(CACHE_REPO):$(TAG) $(MAKE) -C . set-registry-image
	@kubectl get deployment $(CACHE_DEPLOYMENT) -o json | grep -v last | grep -qo debug.activate || DEPLOYMENT=$(CACHE_DEPLOYMENT) $(MAKE) -C . add-debug-flag-agent
	DEPLOYMENT=$(CACHE_DEPLOYMENT) $(MAKE) -C . disable-frozen-modules
	DEPLOYMENT=$(CACHE_DEPLOYMENT) REPLICAS=$(CURRENT_CACHE_REPLICAS) make scale

revert-cache: cluster repo-$(CACHE_REPO) delete-$(CACHE_DEPLOYMENT) ## Reverts the CACHE deployment to use the official image
	DEPLOYMENT=$(CACHE_DEPLOYMENT) IMAGE_FULL_REFERENCE=$(CONTAINER_REGISTRY_BASE)/$(CACHE_REPO):$(FARMVIBES_AI_IMAGE_TAG) make set-registry-image
	DEPLOYMENT=$(CACHE_DEPLOYMENT) REPLICAS=$(CURRENT_CACHE_REPLICAS) make scale

local-rest-api-orchestrator: cluster services-base
	$(eval export PATH=$(HOME)/.config/farmvibes-ai:$(PATH))
	docker build -t $(REST_API_REPO):$(TAG) -t $(ORCHESTRATOR_REPO):$(TAG) -f $(ROOT)/resources/docker/Dockerfile-api_orchestrator .

local-cache-repo: cluster services-base
	$(eval export PATH=$(HOME)/.config/farmvibes-ai:$(PATH))
	docker build -t $(CACHE_REPO):$(TAG) -f $(ROOT)/resources/docker/Dockerfile-cache .

local-worker-repo: cluster worker-base
	$(eval export PATH=$(HOME)/.config/farmvibes-ai:$(PATH))
	docker build -t $(WORKER_REPO):$(TAG) -f $(ROOT)/resources/docker/Dockerfile-worker .

debug-rest-api: cluster local-rest-api  ## Starts listening to debug the REST API
	DEPLOYMENT=$(REST_API_DEPLOYMENT) REPLICAS=1 make scale
	kubectl port-forward deployments/$(REST_API_DEPLOYMENT) $(REST_API_DEBUG_PORT):$(CONTAINER_DEBUG_PORT)

debug-orchestrator: cluster local-orchestrator  ## Starts listening to debug the ORCHESTRATOR
	DEPLOYMENT=$(ORCHESTRATOR_DEPLOYMENT) REPLICAS=1 make scale
	kubectl port-forward deployments/$(ORCHESTRATOR_DEPLOYMENT) $(ORCHESTRATOR_DEBUG_PORT):$(CONTAINER_DEBUG_PORT)

debug-worker: cluster local-worker  ## Starts listening to debug the WORKER
	@kubectl get deployment $(WORKER_DEPLOYMENT) -o json | grep -v last | grep -qo debug.activate || DEPLOYMENT=$(WORKER_DEPLOYMENT) $(MAKE) -C . add-debug-flag-agent
	DEPLOYMENT=$(WORKER_DEPLOYMENT) REPLICAS=1 make scale
	kubectl port-forward pod/`kubectl get pods -l app=$(WORKER_DEPLOYMENT) --field-selector status.phase=Running | awk '/Running/{ print $$1 }'` \
		$(WORKER_DEBUG_PORT):$(CONTAINER_DEBUG_PORT)

debug-cache: cluster local-cache  ## Starts listening to debug the CACHE
	DEPLOYMENT=$(CACHE_DEPLOYMENT) REPLICAS=1 make scale
	kubectl port-forward pod/`kubectl get pods -l app=$(CACHE_DEPLOYMENT) --field-selector status.phase=Running | awk '/Running/{ print $$1 }'` \
		$(CACHE_DEBUG_PORT):$(CONTAINER_DEBUG_PORT)

debug-data-ops: cluster local-data-ops  ## Starts listening to debug the DATA_OPS
	DEPLOYMENT=$(DATA_OPS_DEPLOYMENT) REPLICAS=1 make scale
	kubectl port-forward deployments/$(DATA_OPS_DEPLOYMENT) $(DATA_OPS_DEBUG_PORT):$(CONTAINER_DEBUG_PORT)

clean: cluster revert clean-worker clean-orchestrator clean-rest-api clean-cache

clean-cache: cluster revert-cache revert-worker ## Cleans up the cache image from the local docker "registry"
	docker images | grep -E "$(CACHE_REPO)\\s+tmp.*" | awk '{ print $$3 }' | xargs docker rmi

clean-worker: cluster revert-cache revert-worker ## Cleans up the worker image from the local docker "registry"
	docker images | grep -E "$(WORKER_REPO)\\s+tmp.*" | awk '{ print $$3 }' | xargs docker rmi

clean-orchestrator: cluster revert-rest-api revert-orchestrator ## Cleans up the orchestrator image from the local docker "registry"
	docker images | grep -E "$(ORCHESTRATOR_REPO)\\s+tmp.*" | awk '{ print $$3 }' | xargs docker rmi

clean-data-ops: cluster revert-rest-api revert-data-ops ## Cleans up the data-ops image from the local docker "registry"
	docker images | grep -E "$(DATA_OPS_REPO)\\s+tmp.*" | awk '{ print $$3 }' | xargs docker rmi

clean-rest-api: cluster revert-rest-api revert-orchestrator ## Cleans up the orchestrator image from the local docker "registry"
	docker images | grep -E "$(REST_API_REPO)\\s+tmp.*" | awk '{ print $$3 }' | xargs docker rmi

cluster:
	$(eval export PATH=$(HOME)/.config/farmvibes-ai:$(PATH))
	which k3d || $(build_cluster)
	docker ps | grep -q farmvibes-ai || farmvibes-ai local start || $(build_cluster)
