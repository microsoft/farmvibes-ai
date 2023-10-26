locals {
  service_name = "terravibes-data-ops"
  data_ops_common_args = [
    "data_ops=${var.startup_type}",
    "data_ops.impl.port=3000",
  ]
  data_ops_extra_args = [
    "data_ops.impl.loglevel=${var.farmvibes_log_level}",
    "data_ops.impl.logdir=${var.log_dir}",
    "data_ops.impl.storage.local_path=/mnt/data/stac",
    "data_ops.impl.storage.asset_manager.local_storage_path=/mnt/data/assets",
  ]
}

resource "kubernetes_deployment" "dataops" {
  metadata {
    name      = local.service_name
    namespace = var.namespace
    labels = {
      app     = local.service_name
      backend = "terravibes"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = local.service_name
      }
    }

    template {
      metadata {
        labels = {
          app = local.service_name
        }
        annotations = {
          "dapr.io/enabled"        = "true"
          "dapr.io/app-id"         = local.service_name
          "dapr.io/app-port"       = "3000"
          "dapr.io/app-protocol"   = "http"
          "dapr.io/config"         = "appconfig"
          "dapr.io/enable-metrics" = "true"
          "dapr.io/metrics-port"   = "9090"
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
          name        = local.service_name
          working_dir = var.working_dir
          security_context {
            run_as_user  = var.run_as_user_id
            run_as_group = var.run_as_group_id
          }
          port {
            container_port = 3000
          }
          command = [
            "/opt/conda/bin/vibe-data-ops"
          ]
          args = flatten([
            local.data_ops_common_args, var.local_deployment ? local.data_ops_extra_args : []
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
          env {
            name  = "DAPR_API_METHOD_INVOCATION_PROTOCOL"
            value = "HTTP"
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
