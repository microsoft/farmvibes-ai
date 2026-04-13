# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: Bitnami RabbitMQ images and Helm chart are no longer available on
# Docker Hub (removed in Bitnami's 2025 migration). The Bitnami chart's init
# containers expect Bitnami-specific paths (/opt/bitnami/rabbitmq/) that are
# incompatible with official RabbitMQ images. This module uses native Kubernetes
# resources with the official rabbitmq image instead.
# See: https://github.com/microsoft/farmvibes-ai/issues/234

resource "random_password" "rabbitmq_password" {
  length  = 24
  special = false
}

resource "kubernetes_secret" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  data = {
    "rabbitmq-password" = random_password.rabbitmq_password.result
  }
}

resource "kubernetes_stateful_set" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  spec {
    service_name = "rabbitmq"
    replicas     = 1

    selector {
      match_labels = {
        app = "rabbitmq"
      }
    }

    template {
      metadata {
        labels = {
          app = "rabbitmq"
        }
      }

      spec {
        container {
          name  = "rabbitmq"
          image = "rabbitmq:${var.rabbitmq_image_tag}"

          port {
            container_port = 5672
            name           = "amqp"
          }

          port {
            container_port = 15672
            name           = "management"
          }

          env {
            name  = "RABBITMQ_DEFAULT_USER"
            value = "user"
          }

          env {
            name = "RABBITMQ_DEFAULT_PASS"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.rabbitmq.metadata[0].name
                key  = "rabbitmq-password"
              }
            }
          }

          env {
            name  = "RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS"
            value = "-rabbit consumer_timeout 10800000"
          }

          readiness_probe {
            tcp_socket {
              port = 5672
            }
            initial_delay_seconds = 20
            period_seconds        = 10
          }

          liveness_probe {
            tcp_socket {
              port = 5672
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  spec {
    selector = {
      app = "rabbitmq"
    }

    port {
      name        = "amqp"
      port        = 5672
      target_port = 5672
    }

    port {
      name        = "management"
      port        = 15672
      target_port = 15672
    }
  }
}

data "kubernetes_service" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  depends_on = [kubernetes_service.rabbitmq]
}
