# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "helm_release" "redis" {
  name = "redis"

  repository = "https://charts.bitnami.com/bitnami"
  chart      = "redis"
  namespace  = var.namespace

  set {
    name  = "commonConfiguration"
    value = "appendonly no"
  }

  set {
    name  = "master.containerPort"
    value = "6379"
  }

  set {
    name  = "image.tag"
    value = var.redis_image_tag
  }

  set {
    name  = "replica.replicaCount"
    value = "0"
  }
}

data "kubernetes_service" "redis" {
  metadata {
    name      = "redis-master"
    namespace = var.namespace
  }

  depends_on = [helm_release.redis]
}
