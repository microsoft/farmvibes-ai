resource "kubernetes_config_map" "otel" {
  count = var.enable_telemetry ? 1 : 0
  metadata {
    name = "otel-collector-config"
    labels = {
      app       = "opentelemetry"
      component = "otel-collector-conf"
    }
  }

  data = {
    "otel-collector-config.yaml" = <<EOF
      receivers:
        otlp:
          protocols:
            grpc:
            http:
      processors:
        batch:
      extensions:
        health_check:
        pprof:
          endpoint: :1888
        zpages:
          endpoint: :55679
      exporters:
        debug:
          verbosity: detailed
        azuremonitor:
          endpoint: "https://eastus-8.in.applicationinsights.azure.com/v2/track"
          instrumentation_key: $MONITOR_INSTRUMENTATION_KEY
          # maxbatchsize is the maximum number of items that can be
          # queued before calling to the configured endpoint
          maxbatchsize: 100
          # maxbatchinterval is the maximum time to wait before calling
          # the configured endpoint.
          maxbatchinterval: 10s
      service:
        extensions: [pprof, zpages, health_check]
        pipelines:
          traces:
            receivers: [otlp]
            exporters: [debug, azuremonitor]
    EOF
  }

  depends_on = [
    kubectl_manifest.keyvaultsidecar,
    kubectl_manifest.statestore-sidecar,
    kubectl_manifest.control-pubsub-sidecar
  ]
}

resource "kubernetes_deployment" "otel-collector" {
  count = var.enable_telemetry ? 1 : 0
  metadata {
    name = "otel-collector"
    labels = {
      app       = "otel-collector"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "otel-collector"
      }
    }

    template {
      metadata {
        labels = {
          app       = "otel-collector"
        }
      }

      spec {
        node_selector = {
          agentpool = "default"
        }
        container {
          image = "otel/opentelemetry-collector-contrib:0.87.0"
          name  = "otel-collector"
          port {
            container_port = 4317
          }
          port {
            container_port = 55681
          }
          port {
            container_port = 13133
          }
          resources {
            limits = {
              cpu    = "1"
              memory = "2Gi"
            }
            requests = {
              cpu    = "200m"
              memory = "400Mi"
            }
          }
          liveness_probe {
            http_get {
              path = "/"
              port = 13133
            }
          }
          readiness_probe {
            http_get {
              path = "/"
              port = 13133
            }
          }
          volume_mount {
            name      = "otel-collector-config-vol"
            mount_path = "/conf"
          }

          env {
            name = "MONITOR_INSTRUMENTATION_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.monitor_instrumentation_key_secret.metadata[0].name
                key  = "monitor_instrumentation_key"
              }
            }
          }

          args = ["--config=/conf/otel-collector-config.yaml"]
        }

        volume {
          name = "otel-collector-config-vol"

          config_map {
            name = kubernetes_config_map.otel[0].metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "otel_collector" {
  count = var.enable_telemetry ? 1 : 0
  metadata {
    name = "otel-collector"
    labels = {
      app       = "opencesus"
      component = "otel-collector"
    }
  }

  spec {
    selector = {
      app = "otel-collector"
    }

    port {
      name       = "otlp"
      port       = 4317
      target_port = 4317
      protocol   = "TCP"
    }

    port {
      name       = "metrics"
      port       = 55681
      target_port = 55681
      protocol   = "TCP"
    }
  }

  depends_on = [
    kubernetes_deployment.otel-collector
  ]
}

locals {
  otel_name      = var.enable_telemetry ? kubernetes_service.otel_collector[0].metadata[0].name : ""
  otel_namespace = var.enable_telemetry ? kubernetes_service.otel_collector[0].metadata[0].namespace : ""
  otel_port      = var.enable_telemetry ? kubernetes_service.otel_collector[0].spec[0].port[0].port : ""
}

output "otel_service_name" {
  value = var.enable_telemetry ? "http://${local.otel_name}.${local.otel_namespace}.svc.cluster.local:${local.otel_port}" : ""

  depends_on = [
    kubernetes_service.otel_collector
  ]
}