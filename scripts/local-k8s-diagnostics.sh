#!/bin/sh

PATH=$PATH:~/.config/farmvibes-ai

echo "kubectl location:"
which kubectl

echo "Cluster pods:"
kubectl get pods
kubectl get pods -o yaml

echo "Docker images:"
docker images

echo "REST API description:"
kubectl describe deployment terravibes-rest-api

echo "Orchestrator description:"
kubectl describe deployment terravibes-orchestrator

echo "Worker description:"
kubectl describe deployment terravibes-worker

echo "Cache description:"
kubectl describe deployment terravibes-cache

echo "REST API logs:"
kubectl logs -l app=terravibes-rest-api --all-containers=true --tail=-1

echo "Orchestrator logs:"
kubectl logs -l app=terravibes-orchestrator --all-containers=true --tail=-1

echo "Worker logs:"
kubectl logs -l app=terravibes-worker --max-log-requests=8 --all-containers=true --tail=-1

echo "Cache logs:"
kubectl logs -l app=terravibes-cache --all-containers=true --tail=-1

echo "Data Ops logs:"
kubectl logs -l app=terravibes-data-ops --all-containers=true --tail=-1

echo "Kubernetes logs:"
docker ps | egrep 'k3d-farmvibes-ai-.*-0' | awk '{ print $1 }' | xargs docker logs
