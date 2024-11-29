# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

locals {
  orchestrator_common_args = concat(
    [
      "-Xfrozen_modules=on",
      "/opt/conda/bin/vibe-orchestrator",
      "--port=3000",
    ],
    var.otel_service_name != "" ? [
      "--otel-service-name=${var.otel_service_name}",
    ] : []
  )
  orchestrator_extra_args = concat(
    [
      "--logdir=${var.log_dir}",
      "--loglevel=${var.farmvibes_log_level}",
    ],
    var.max_log_file_bytes != "" ? [
      "--max-log-file-bytes=${var.max_log_file_bytes}"
    ] : [],
    var.log_backup_count != "" ? [
      "--log-backup-count=${var.log_backup_count}"
    ] : [],
  )
}

resource "kubernetes_deployment" "orchestrator" {
  metadata {
    name      = "terravibes-orchestrator"
    namespace = var.namespace
    labels = {
      app     = "terravibes-orchestrator"
      backend = "terravibes"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "terravibes-orchestrator"
      }
    }

    template {
      metadata {
        labels = {
          app = "terravibes-orchestrator"
        }
        annotations = {
          "dapr.io/enabled"        = "true"
          "dapr.io/app-id"         = "terravibes-orchestrator"
          "dapr.io/app-port"       = "3000"
          "dapr.io/app-protocol"   = "http"
          "dapr.io/config"         = "appconfig"
          "dapr.io/enable-metrics" = "true"
          "dapr.io/metrics-port"   = "9090"
          "dapr.io/log-as-json"    = "true"
          "prometheus.io/scrape"   = "true"
          "prometheus.io/port"     = "9090"
          "prometheus.io/path"     = "/"
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
          image       = "${var.acr_registry}/${var.image_prefix}api-orchestrator:${var.image_tag}"
          name        = "terravibes-orchestrator"
          working_dir = var.working_dir
          security_context {
            run_as_user  = var.run_as_user_id
            run_as_group = var.run_as_group_id
          }
          port {
            container_port = 3000
          }
          command = [
            "/opt/conda/bin/python",
          ]
          args = flatten([
            local.orchestrator_common_args, var.local_deployment ? local.orchestrator_extra_args : []
          ])
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
