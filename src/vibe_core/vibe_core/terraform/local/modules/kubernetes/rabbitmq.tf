# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "helm_release" "rabbitmq" {
  name = "rabbitmq"

  repository = "oci://registry-1.docker.io/bitnamicharts"
  chart      = "rabbitmq"
  namespace  = var.namespace

  set {
    name  = "image.tag"
    value = var.rabbitmq_image_tag
  }

  set {
    name  = "containerPorts.amqp"
    value = "5672"
  }

  set {
    name  = "containerPorts.amqpTls"
    value = "5671"
  }

  set {
    name  = "containerPorts.dist"
    value = "25672"
  }

  set {
    name  = "containerPorts.manager"
    value = "15672"
  }

  set {
    name  = "containerPorts.epmd"
    value = "4369"
  }

  set {
    name  = "containerPorts.metrics"
    value = "9419"
  }

  set {
    name  = "replica.replicaCount"
    value = "1"
  }

  set {
    name  = "extraEnvVars[0].name"
    value = "RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS"
  }

  set {
    name  = "extraEnvVars[0].value"
    value = "-rabbit consumer_timeout 10800000"
  }
}

data "kubernetes_service" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  depends_on = [helm_release.rabbitmq]
}
