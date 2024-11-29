locals {
  cache_common_args = concat(
      [
        "cache=${var.startup_type}",
        "cache.impl.port=3000"
      ], 
      var.otel_service_name != "" ? [
        "cache.impl.otel_service_name=${var.otel_service_name}"
      ] : []
    )

  cache_extra_args = concat(
    [
      "cache.impl.loglevel=${var.farmvibes_log_level}",
      "cache.impl.logdir=${var.log_dir}",
    ],
    var.max_log_file_bytes != "" ? [
      "cache.impl.max_log_file_bytes=${var.max_log_file_bytes}",
    ] : [],
    var.log_backup_count != "" ? [
      "cache.impl.log_backup_count=${var.log_backup_count}",
    ] : [],
  )
}

resource "kubernetes_deployment" "cache" {
  metadata {
    name      = "terravibes-cache"
    namespace = var.namespace
    labels = {
      app     = "terravibes-cache"
      backend = "terravibes"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "terravibes-cache"
      }
    }

    template {
      metadata {
        labels = {
          app = "terravibes-cache"
        }
        annotations = {
          "dapr.io/enabled"        = "true"
          "dapr.io/app-id"         = "terravibes-cache"
          "dapr.io/app-port"       = "3000"
          "dapr.io/app-protocol"   = "grpc"
          "dapr.io/enable-metrics" = "true"
          "dapr.io/metrics-port"   = "9090"
          "dapr.io/log-as-json"    = "true"
        }
      }

      spec {
        node_selector = {
          agentpool = var.default_node_pool_name
        }
        image_pull_secrets {
          name = "acrtoken"
        }
        container {
          image       = "${var.acr_registry}/${var.image_prefix}cache:${var.image_tag}"
          name        = "terravibes-cache"
          working_dir = var.working_dir
          security_context {
            run_as_user  = var.run_as_user_id
            run_as_group = var.run_as_group_id
          }
          port {
            container_port = 3000
          }
          command = [
            "/opt/conda/bin/vibe-cache",
          ]
          args = flatten([
            local.cache_common_args, var.local_deployment ? local.cache_extra_args : []
          ])
          env {
            name = "AZURE_TENANT_ID"
            value_from {
              secret_key_ref {
                name     = "service-principal-secret"
                key      = "tenant"
                optional = true
              }
            }
          }
          env {
            name = "AZURE_CLIENT_ID"
            value_from {
              secret_key_ref {
                name     = "service-principal-secret"
                key      = "client"
                optional = true
              }
            }
          }
          env {
            name = "AZURE_CLIENT_SECRET"
            value_from {
              secret_key_ref {
                name     = "service-principal-secret"
                key      = "password"
                optional = true
              }
            }
          }
          env {
            name  = "BLOB_CONTAINER_NAME"
            value = "assets"
          }
          env {
            name  = "BLOB_STORAGE_ACCOUNT_CONNECTION_STRING"
            value = "storage-account-connection-string"
          }
          env {
            name  = "STAC_COSMOS_URI_SECRET"
            value = "stac-cosmos-db-url"
          }
          env {
            name  = "STAC_CONTAINER_NAME_SECRET"
            value = "stac-cosmos-container-name"
          }
          env {
            name  = "STAC_ASSETS_CONTAINER_NAME_SECRET"
            value = "stac-cosmos-assets-container-name"
          }
          env {
            name  = "STAC_COSMOS_DATABASE_NAME_SECRET"
            value = "stac-cosmos-db-name"
          }
          env {
            name  = "STAC_COSMOS_CONNECTION_KEY_SECRET"
            value = "stac-cosmos-write-key"
          }
          dynamic "volume_mount" {
            for_each = var.local_deployment ? [1] : []
            content {
              mount_path = "/mnt/"
              name       = "host-mount"
            }
          }
        }
        dynamic "volume" {
          for_each = var.local_deployment ? [1] : []
          content {
            host_path {
              path = "/mnt/"
            }
            name = "host-mount"
          }
        }
      }
    }
  }

  depends_on = [
    var.dapr_sidecars_deployed
  ]
}
