locals {
  worker_common_args = [
    "worker=${var.startup_type}",
    "worker.impl.control_topic=commands",
    "worker.impl.port=3000",
  ]
  worker_extra_args = [
    "worker.impl.logdir=${var.log_dir}",
    "worker.impl.loglevel=${var.farmvibes_log_level}",
    "worker.impl.factory_spec.storage.local_path=/mnt/data/stac",
    "worker.impl.factory_spec.storage.asset_manager.local_storage_path=/mnt/data/assets",
  ]
}

resource "kubernetes_deployment" "worker" {
  metadata {
    name      = "terravibes-worker"
    namespace = var.namespace
    labels = {
      app     = "terravibes-worker"
      backend = "terravibes"
    }
  }

  spec {
    replicas = var.worker_replicas

    selector {
      match_labels = {
        app = "terravibes-worker"
      }
    }

    template {
      metadata {
        labels = {
          app = "terravibes-worker"
        }
        annotations = {
          "dapr.io/enabled"        = "true"
          "dapr.io/app-id"         = "terravibes-worker"
          "dapr.io/app-port"       = "3000"
          "dapr.io/config"         = "appconfig"
          "dapr.io/app-protocol"   = "http"
          "dapr.io/enable-metrics" = "true"
          "dapr.io/metrics-port"   = "9090"
          "prometheus.io/scrape"   = "true"
          "prometheus.io/port"     = "9090"
          "prometheus.io/path"     = "/"
        }
      }

      spec {
        node_selector = {
          agentpool = var.worker_node_pool_name
        }
        image_pull_secrets {
          name = "acrtoken"
        }
        container {
          image = "${var.acr_registry}/${var.image_prefix}worker:${var.image_tag}"
          name  = "terravibes-worker"
          working_dir = var.working_dir
          security_context {
            run_as_user = var.run_as_user_id
            run_as_group = var.run_as_group_id
          }
          port {
            container_port = 3000
          }
          lifecycle {
            pre_stop {
                exec {
                    command = ["/usr/bin/curl", "http://localhost:3500/v1.0/invoke/terravibes-worker/method/shutdown"]
                }
            }
          }
          command = [
            "/opt/conda/bin/vibe-worker",
          ]
          args = flatten([
            local.worker_common_args, var.local_deployment ? local.worker_extra_args : []
          ])
          resources {
            limits = {
              memory = "32Gi"
            }
            requests = {
              memory = var.worker_memory_request
            }
          }
          env {
            name  = "HOME"
            value = "/tmp"
          }
          env {
            name  = "DAPR_API_METHOD_INVOCATION_PROTOCOL"
            value = "HTTP"
          }
          env {
            name = "AZURE_TENANT_ID"
            value_from {
              secret_key_ref {
                name = "service-principal-secret"
                key  = "tenant"
                optional = true
              }
            }
          }
          env {
            name = "AZURE_CLIENT_ID"
            value_from {
              secret_key_ref {
                name = "service-principal-secret"
                key  = "client"
                optional = true
              }
            }
          }
          env {
            name = "AZURE_CLIENT_SECRET"
            value_from {
              secret_key_ref {
                name = "service-principal-secret"
                key  = "password"
                optional = true
              }
            }
          }
          env {
            name = "BLOB_CONTAINER_NAME"
            value = "assets-fan"
          }
          env {
            name = "BLOB_STORAGE_ACCOUNT_CONNECTION_STRING"
            value = "storage-account-connection-string"
          }
          env {
            name = "STAC_COSMOS_URI_SECRET"
            value = "stac-cosmos-db-url"
          }
          env {
            name = "STAC_CONTAINER_NAME_SECRET"
            value = "stac-cosmos-container-name"
          }
          env {
            name = "STAC_ASSETS_CONTAINER_NAME_SECRET"
            value = "stac-cosmos-assets-container-name"
          }
          env {
            name = "STAC_COSMOS_DATABASE_NAME_SECRET"
            value = "stac-cosmos-db-name"
          }
          env {
            name = "STAC_COSMOS_CONNECTION_KEY_SECRET"
            value = "stac-cosmos-write-key"
          }
          volume_mount {
            mount_path = "/mnt"
            name = "shared-resources"
          }
          dynamic "volume_mount" {
            for_each = var.local_deployment ? [] : [1]
            content {
              mount_path = "/dev/shm"
              name = "dshm"
            }
          }
        }
        volume {
          name = "shared-resources"
          persistent_volume_claim {
            claim_name = var.shared_resource_pv_claim_name
          }
        }
        dynamic "volume" {
          for_each = var.local_deployment ? [] : [1]
          content {
            empty_dir {
              medium = "Memory"
              size_limit = "2Gi"
            }
            name = "dshm"
          }  
        }
      }
    }
  }

  timeouts {
    create = "20m"
    update = "20m"
  }

  depends_on = [
    var.dapr_sidecars_deployed
  ]
}
