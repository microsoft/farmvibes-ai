resource "kubernetes_config_map" "otel_collector_config" {
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
      exporters:
        debug:
          verbosity: detailed
        otlphttp:
          endpoint: "http://jaeger-collector.default.svc.cluster.local:4318"
      service:
        pipelines:
          traces:
            receivers: [otlp]
            exporters: [debug, otlphttp]
    EOF
  }
}

resource "kubernetes_deployment" "otel_collector_deployment" {
  count = var.enable_telemetry ? 1 : 0
  metadata {
    name = "otel-collector"
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
          app = "otel-collector"
        }
      }

      spec {
        container {
          name  = "otel-collector"
          image = "mcr.microsoft.com/oss/otel/opentelemetry-collector:0.91.0"

          port {
            container_port = 4317
          }

          port {
            container_port = 55681
          }

          volume_mount {
            name      = "otel-collector-config-vol"
            mount_path = "/conf"
          }

          args = ["--config=/conf/otel-collector-config.yaml"]
        }

        volume {
          name = "otel-collector-config-vol"

          config_map {
            name = kubernetes_config_map.otel_collector_config[0].metadata[0].name
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