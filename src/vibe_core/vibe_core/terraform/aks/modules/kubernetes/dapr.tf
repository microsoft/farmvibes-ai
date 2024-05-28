resource "kubernetes_namespace" "kubernetesdaprnamespace" {
  metadata {
    name = "dapr-system"
  }
}

variable "dapr_cloud_environment" {
  type = map
  default = {
    "public" = "AZUREPUBLICCLOUD"
    "china" = "AZURECHINACLOUD"
    "german" = "AZUREGERMANCLOUD"
    "usgovernment" = "AZUREUSGOVERNMENTCLOUD"
  }
}

resource "helm_release" "dapr" {
  name       = "dapr"
  repository = "https://dapr.github.io/helm-charts/"
  chart      = "dapr"
  namespace  = "dapr-system"
  version    = "1.13.3"

  set {
    name  = "enable-ha"
    value = "true"
  }

  depends_on = [helm_release.letsencrypt, kubernetes_namespace.kubernetesdaprnamespace]
}

resource "kubectl_manifest" "keyvaultsidecar" {
  yaml_body = <<-EOF
    apiVersion: dapr.io/v1alpha1
    kind: Component
    metadata:
      name: azurekeyvault
      namespace: ${var.namespace}
    spec:
      type: secretstores.azure.keyvault
      version: v1
      metadata:
      - name: vaultName
        value: ${var.keyvault_name}
      - name: azureClientId
        value: ${var.application_id}
      - name: azureEnvironment
        value: ${var.dapr_cloud_environment[var.environment]}
    EOF

  depends_on = [helm_release.dapr]
}

resource "kubectl_manifest" "control-pubsub-sidecar" {
  yaml_body = <<-EOF
    apiVersion: dapr.io/v1alpha1
    kind: Component
    metadata:
      name: control-pubsub
      namespace: ${var.namespace}
    spec:
      type: pubsub.rabbitmq
      version: v1
      metadata:
      - name: protocol
        value: amqp
      - name: hostname
        value: ${data.kubernetes_service.rabbitmq.metadata.0.name}.${var.namespace}.svc.cluster.local
      - name: port
        value: 5672
      - name: deleteWhenUnused
        value: "true"
      - name: requeueInFailure
        value: "true"
      - name: prefetchCount
        value: "1"
      - name: publisherConfirm
        value: "true"
      - name: durable
        value: "true"
      - name: deliveryMode
        value: "2"
      - name: password
        secretKeyRef:
          name: rabbitmq
          key: rabbitmq-password
      - name: username
        value: user
    EOF

  depends_on = [helm_release.dapr, data.kubernetes_service.rabbitmq]
}

resource "kubectl_manifest" "statestore-sidecar" {
  yaml_body = <<-EOF
    apiVersion: dapr.io/v1alpha1
    kind: Component
    metadata:
      name: statestore
      namespace: ${var.namespace}
    spec:
      type: state.azure.cosmosdb
      version: v1
      metadata:
      - name: url
        secretKeyRef:
          name: cosmos-db-url
          key: cosmos-db-url
      - name: masterKey
        secretKeyRef:
          name: cosmos-db-key
          key: cosmos-db-key
      - name: database
        secretKeyRef:
          name: cosmos-db-database
          key: cosmos-db-database
      - name: collection
        secretKeyRef:
          name: cosmos-db-collection
          key: cosmos-db-collection
      - name: keyPrefix
        value: none
    auth:
      secretStore: azurekeyvault
    EOF

  depends_on = [helm_release.dapr, kubectl_manifest.keyvaultsidecar]
}

resource "kubectl_manifest" "resiliency-sidecar" {
  yaml_body = <<-EOF
    apiVersion: dapr.io/v1alpha1
    kind: Resiliency
    metadata:
      name: worker-resiliency
    scopes:
      - terravibes-worker
    spec:
      policies:
        timeouts:
          opExecution: 3h  # should be bigger than any individual op run
        retries:
          workerRetry:
            policy: exponential
            maxInterval: 60s
            maxRetries: -1
      targets:
        components:
          control-pubsub:
            inbound:
              retry: "workerRetry"
              timeout: "opExecution"
    EOF

  depends_on = [helm_release.dapr, kubectl_manifest.statestore-sidecar]
}

resource "kubectl_manifest" "daprconfigcollector" {
  yaml_body = <<-EOF
    apiVersion: dapr.io/v1alpha1
    kind: Configuration
    metadata:
      name: appconfig
      namespace: ${var.namespace}
    spec:
      tracing:
        samplingRate: "1"
        zipkin:
          endpointAddress: "http://otel-collector.default.svc.cluster.local:9411/api/v2/spans"
    EOF

  depends_on = [helm_release.dapr, kubernetes_namespace.kubernetesnamespace]
}
