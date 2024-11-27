# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "kubernetes_deployment" "jaeger" {
  count = var.enable_telemetry ? 1 : 0
  metadata {
    name = "jaeger"
    namespace = var.namespace
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "jaeger"
      }
    }

    template {
      metadata {
        labels = {
          app = "jaeger"
        }
      }

      spec {
        container {
          image = "mcr.microsoft.com/oss/jaegertracing/all-in-one:v1.49.0-2"
          name  = "jaeger"
          command = ["/go/bin/all-in-one-linux"]

          port {
            container_port = 14250
            protocol       = "TCP"
          }

          port {
            container_port = 14268
            protocol       = "TCP"
          }

          port {
            container_port = 16686
            protocol       = "TCP"
          }

          port {
            container_port = 4318
            protocol       = "TCP"
          }

          env {
            name  = "COLLECTOR_OTLP_ENABLED"
            value = "true"
          }

          readiness_probe {
            http_get {
              path = "/"
              port = 14269
            }
          }

          liveness_probe {
            http_get {
              path = "/"
              port = 14269
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "jaeger_query" {
  count = var.enable_telemetry ? 1 : 0
  metadata {
    name = "jaeger-query"
    namespace = var.namespace
  }

  spec {
    port {
      name       = "query-http"
      port       = 16686
      protocol   = "TCP"
      target_port = 16686
    }

    selector = {
      app = "jaeger"
    }

    session_affinity = "None"
    type             =  "ClusterIP"
  }

  depends_on = [kubernetes_deployment.jaeger]
}

resource "kubernetes_service" "jaeger_collector" {
  count = var.enable_telemetry ? 1 : 0
  metadata {
    name = "jaeger-collector"
    namespace = var.namespace
  }

  spec {
    port {
      name       = "collector-grpc"
      port       = 14250
      protocol   = "TCP"
      target_port = 14250
    }

    port {
      name       = "otlp-collector-http"
      port       = 4318
      protocol   = "TCP"
      target_port = 4318
    }

    port {
      name       = "collector-http"
      port       = 14268
      protocol   = "TCP"
      target_port = 14268
    }

    selector = {
      app = "jaeger"
    }

    session_affinity = "None"
    type             =  "ClusterIP"
  }

  depends_on = [kubernetes_deployment.jaeger]
}
