# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

locals {
  rabbitmq_labels = {
    app = "rabbitmq"
  }
}

resource "random_password" "rabbitmq" {
  length  = 32
  special = false
}

resource "kubernetes_secret" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  data = {
    rabbitmq-password = random_password.rabbitmq.result
  }
}

resource "kubernetes_service" "rabbitmq_headless" {
  metadata {
    name      = "rabbitmq-headless"
    namespace = var.namespace
  }

  spec {
    cluster_ip                  = "None"
    publish_not_ready_addresses = true
    selector                    = local.rabbitmq_labels

    port {
      name        = "epmd"
      port        = 4369
      target_port = "epmd"
    }

    port {
      name        = "amqp"
      port        = 5672
      target_port = "amqp"
    }

    port {
      name        = "dist"
      port        = 25672
      target_port = "dist"
    }

    port {
      name        = "http-stats"
      port        = 15672
      target_port = "management"
    }
  }
}

resource "kubernetes_service" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  spec {
    selector = local.rabbitmq_labels

    port {
      name        = "amqp"
      port        = 5672
      target_port = "amqp"
    }

    port {
      name        = "dist"
      port        = 25672
      target_port = "dist"
    }

    port {
      name        = "http-stats"
      port        = 15672
      target_port = "management"
    }

    port {
      name        = "epmd"
      port        = 4369
      target_port = "epmd"
    }
  }
}

resource "kubernetes_stateful_set" "rabbitmq" {
  metadata {
    name      = "rabbitmq"
    namespace = var.namespace
  }

  spec {
    replicas     = 1
    service_name = kubernetes_service.rabbitmq_headless.metadata[0].name

    selector {
      match_labels = local.rabbitmq_labels
    }

    template {
      metadata {
        labels = local.rabbitmq_labels
      }

      spec {
        automount_service_account_token  = false
        termination_grace_period_seconds = 120

        image_pull_secrets {
          name = "acrtoken"
        }

        container {
          name              = "rabbitmq"
          image             = var.rabbitmq_image
          image_pull_policy = "IfNotPresent"

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

          port {
            name           = "amqp"
            container_port = 5672
          }

          port {
            name           = "dist"
            container_port = 25672
          }

          port {
            name           = "management"
            container_port = 15672
          }

          port {
            name           = "epmd"
            container_port = 4369
          }

          volume_mount {
            name       = "rabbitmq-data"
            mount_path = "/var/lib/rabbitmq"
          }

          startup_probe {
            failure_threshold = 18
            period_seconds    = 10
            timeout_seconds   = 10

            exec {
              command = ["rabbitmq-diagnostics", "-q", "ping"]
            }
          }

          readiness_probe {
            failure_threshold = 3
            period_seconds    = 10
            timeout_seconds   = 10

            exec {
              command = ["rabbitmq-diagnostics", "-q", "check_running"]
            }
          }

          liveness_probe {
            failure_threshold     = 6
            initial_delay_seconds = 30
            period_seconds        = 30
            timeout_seconds       = 10

            exec {
              command = ["rabbitmq-diagnostics", "-q", "ping"]
            }
          }
        }
      }
    }

    volume_claim_template {
      metadata {
        name = "rabbitmq-data"
      }

      spec {
        access_modes = ["ReadWriteOnce"]

        resources {
          requests = {
            storage = "8Gi"
          }
        }
      }
    }
  }
}
