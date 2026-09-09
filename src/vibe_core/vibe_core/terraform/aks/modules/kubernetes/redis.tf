# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

locals {
  redis_labels = {
    app = "redis"
  }
}

resource "random_password" "redis" {
  length  = 32
  special = false
}

resource "kubernetes_secret" "redis" {
  metadata {
    name      = "redis"
    namespace = var.namespace
  }

  data = {
    redis-password = random_password.redis.result
  }
}

resource "kubernetes_service" "redis_headless" {
  metadata {
    name      = "redis-headless"
    namespace = var.namespace
  }

  spec {
    cluster_ip = "None"
    selector   = local.redis_labels

    port {
      name        = "tcp-redis"
      port        = 6379
      target_port = "redis"
    }
  }
}

resource "kubernetes_service" "redis" {
  metadata {
    name      = "redis-master"
    namespace = var.namespace
  }

  spec {
    selector = local.redis_labels

    port {
      name        = "tcp-redis"
      port        = 6379
      target_port = "redis"
    }
  }
}

resource "kubernetes_stateful_set" "redis" {
  metadata {
    name      = "redis-master"
    namespace = var.namespace
  }

  spec {
    replicas     = 1
    service_name = kubernetes_service.redis_headless.metadata[0].name

    selector {
      match_labels = local.redis_labels
    }

    template {
      metadata {
        labels = local.redis_labels
      }

      spec {
        automount_service_account_token  = false
        termination_grace_period_seconds = 30

        image_pull_secrets {
          name = "acrtoken"
        }

        container {
          name              = "redis"
          image             = var.redis_image
          image_pull_policy = "IfNotPresent"
          args              = ["--appendonly", "no", "--requirepass", "$(REDIS_PASSWORD)"]

          env {
            name = "REDIS_PASSWORD"

            value_from {
              secret_key_ref {
                name = kubernetes_secret.redis.metadata[0].name
                key  = "redis-password"
              }
            }
          }

          env {
            name = "REDISCLI_AUTH"

            value_from {
              secret_key_ref {
                name = kubernetes_secret.redis.metadata[0].name
                key  = "redis-password"
              }
            }
          }

          port {
            name           = "redis"
            container_port = 6379
          }

          volume_mount {
            name       = "redis-data"
            mount_path = "/data"
          }

          startup_probe {
            failure_threshold = 24
            period_seconds    = 5
            timeout_seconds   = 2

            tcp_socket {
              port = 6379
            }
          }

          readiness_probe {
            failure_threshold = 3
            period_seconds    = 5
            timeout_seconds   = 2

            exec {
              command = ["sh", "-c", "test \"$(redis-cli ping)\" = PONG"]
            }
          }

          liveness_probe {
            failure_threshold     = 5
            initial_delay_seconds = 20
            period_seconds        = 10
            timeout_seconds       = 2

            tcp_socket {
              port = 6379
            }
          }
        }
      }
    }

    volume_claim_template {
      metadata {
        name = "redis-data"
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
