resource "kubernetes_config_map" "otel" {
  metadata {
    name = "otel-collector-conf"
    labels = {
      app       = "opentelemetry"
      component = "otel-collector-conf"
    }
  }
  data = {
    "otel-collector-config" = <<EOF
      receivers:
        zipkin:
          endpoint: 0.0.0.0:9411
      extensions:
        health_check:
        pprof:
          endpoint: :1888
        zpages:
          endpoint: :55679
      exporters:
        logging:
          loglevel: debug
        azuremonitor:
          endpoint: "https://eastus-8.in.applicationinsights.azure.com/v2/track"
          instrumentation_key: 7f7e7b9e-7ee4-462b-8e62-d158a234b98c
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
            receivers: [zipkin]
            exporters: [azuremonitor,logging]
    EOF
  }

  depends_on = [
    kubectl_manifest.keyvaultsidecar,
    kubectl_manifest.statestore-sidecar,
    kubectl_manifest.control-pubsub-sidecar
  ]
}

resource "kubernetes_deployment" "otel-collector" {
  metadata {
    name = "otel-collector"
    labels = {
      app       = "opentelemetry"
      component = "otel-collector-conf"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "opentelemetry"
      }
    }

    template {
      metadata {
        labels = {
          app       = "opentelemetry"
          component = "otel-collector-conf"
        }
      }

      spec {
        node_selector = {
          agentpool = "default"
        }
        container {
          image = "otel/opentelemetry-collector-contrib:0.40.0"
          name  = "otel-collector"
          port {
            container_port = 9411
          }
          command = [
            "/otelcontribcol",
            "--config=/conf/otel-collector-config.yaml"
          ]
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
            mount_path = "/conf"
            name       = "otel-collector-config-vol"
          }
        }
        volume {
          name = "otel-collector-config-vol"
          config_map {
            name = "otel-collector-conf"
            items {
              key  = "otel-collector-config"
              path = "otel-collector-config.yaml"
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_config_map.otel
  ]
}

resource "kubernetes_service" "otel-collector" {
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
      port        = 9411
      target_port = 9411
      name        = "zipkin"
      protocol    = "TCP"
    }
  }

  depends_on = [
    kubernetes_deployment.otel-collector
  ]
}