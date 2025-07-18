# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "helm_release" "rabbitmq" {
  name = "rabbitmq"

  repository = "oci://registry-1.docker.io/bitnamicharts"
  chart      = "rabbitmq"
  namespace  = var.namespace

  set = [
    {
      name  = "image.tag"
      value = var.rabbitmq_image_tag
    },
    {
      name  = "containerPorts.amqp"
      value = "5672"
    },
    {
      name  = "containerPorts.amqpTls"
      value = "5671"
    },
    {
      name  = "containerPorts.dist"
      value = "25672"
    },
    {
      name  = "containerPorts.manager"
      value = "15672"
    },
    {
      name  = "containerPorts.epmd"
      value = "4369"
    },
    {
      name  = "containerPorts.metrics"
      value = "9419"
    },
    {
      name  = "replica.replicaCount"
      value = "1"
    },
    {
      name  = "extraEnvVars[0].name"
      value = "RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS"
    },
    {
      name  = "extraEnvVars[0].value"
      value = "-rabbit consumer_timeout 10800000"
    }
  ]
}

data "kubernetes_service" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  depends_on = [helm_release.rabbitmq]
}
