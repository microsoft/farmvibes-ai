# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "helm_release" "redis" {
  name = "redis"

  repository = "oci://registry-1.docker.io/bitnamicharts"
  chart      = "redis"
  namespace  = var.namespace

  set = [
    {
      name  = "commonConfiguration"
      value = "appendonly no"
    },
    {
      name  = "master.containerPort"
      value = "6379"
    },
    {
      name  = "image.tag"
      value = var.redis_image_tag
    },
    {
      name  = "replica.replicaCount"
      value = "0"
    }
  ]
}

data "kubernetes_service" "redis" {
  metadata {
    name      = "redis-master"
    namespace = var.namespace
  }

  depends_on = [helm_release.redis]
}
