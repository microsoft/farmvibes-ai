locals {
  restapi_common_args = [
    "--port=3000",
  ]
  restapi_extra_args = [
    "--logdir=${var.log_dir}",
    "--terravibes-host-assets-dir=${var.host_assets_dir}",
    "--loglevel=${var.farmvibes_log_level}",
  ]
}

resource "kubernetes_deployment" "restapi" {
  metadata {
    name      = "terravibes-rest-api"
    namespace = var.namespace
    labels = {
      app     = "terravibes-rest-api"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "terravibes-rest-api"
      }
    }

    template {
      metadata {
        labels = {
          app = "terravibes-rest-api"
        }
        annotations = {
          "dapr.io/enabled"        = "true"
          "dapr.io/app-id"         = "terravibes-rest-api"
          "dapr.io/app-port"       = "3000"
          "dapr.io/app-protocol"   = "http"
          "dapr.io/config"         = "appconfig"
          "dapr.io/enable-metrics" = "true"
          "dapr.io/metrics-port"   = "9090"
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
          image = "${var.acr_registry}/${var.image_prefix}api-orchestrator:${var.image_tag}"
          name  = "terravibes-rest-api"
          port {
            container_port = 3000
          }
          working_dir = var.working_dir
          security_context {
            run_as_user = var.run_as_user_id
            run_as_group = var.run_as_group_id
          }
          command = [
            "/opt/conda/bin/vibe-server",
          ]
          args = flatten([
            local.restapi_common_args, var.local_deployment ? local.restapi_extra_args : []
          ])
          env {
            name  = "DAPR_API_METHOD_INVOCATION_PROTOCOL"
            value = "HTTP"
          }
          dynamic "volume_mount" {
            for_each = var.local_deployment ? [1] : []
            content {
              mount_path = "/mnt/"
              name = "host-mount"
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

resource "kubernetes_service" "restapi-local" {
  count = var.local_deployment ? 1 : 0
  metadata {
    name      = "terravibes-rest-api"
    namespace = var.namespace
    labels = {
      app     = "terravibes-rest-api"
      backend = "terravibes"
    }
  }
  spec {
    selector = {
      app = "terravibes-rest-api"
    }
    port {
      port        = 30000
      target_port = 3000
      name        = "http"
    }
    type = "NodePort"
  }

  depends_on = [
    kubernetes_deployment.restapi
  ]
}

resource "kubernetes_service" "restapi" {
  count = var.local_deployment ? 0 : 1
  metadata {
    name      = "terravibes-rest-api"
    namespace = var.namespace
    labels = {
      app     = "terravibes-rest-api"
      backend = "terravibes"
    }
  }
  spec {
    selector = {
      app = "terravibes-rest-api"
    }
    port {
      port        = 80
      target_port = 3000
      name        = "tcp"
      protocol    = "TCP"
    }
    type = "ClusterIP"
  }

  depends_on = [
    kubernetes_deployment.restapi
  ]
}

resource "kubernetes_ingress_v1" "restapi" {
  count = var.local_deployment ? 0 : 1
  wait_for_load_balancer = true
  metadata {
    name      = "terravibes-rest-api-ingress"
    namespace = var.namespace
    annotations = {
      "nginx.ingress.kubernetes.io/use-regex"      = "true"
      "nginx.ingress.kubernetes.io/ssl-redirect"   = "false"
    }
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = var.public_ip_fqdn
      http {
        path {
          path = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service.restapi[0].metadata.0.name
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }

  depends_on = [ kubernetes_service.restapi ]
}
