resource "kubernetes_namespace" "kubernetesdaprnamespace" {
  metadata {
    name = "dapr-system"
  }
}

resource "helm_release" "dapr" {
  name       = "dapr"
  repository = "https://dapr.github.io/helm-charts/"
  chart      = "dapr"
  namespace  = "dapr-system"
  version    = "1.13.3"

  set {
    name  = "dapr_operator.watchInterval"
    value = "30s"
  }

  set {
    name  = "enable-ha"
    value = "true"
  }

  depends_on = [kubernetes_namespace.kubernetesdaprnamespace]
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
      - name: password
        secretKeyRef:
          name: rabbitmq
          key: rabbitmq-password
      - name: username
        value: user
      - name: backOffMaxRetries
        value: -1
      - name: publisherConfirm
        value: "true"
      - name: requeueInFailure
        value: "true"
      - name: deliveryMode
        value: "2"
      - name: prefetchCount
        value: 0
      - name: deletedWhenUnused
        value: "false"
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
      type: state.redis
      version: v1
      metadata:
      - name: redisHost
        value: redis-master:6379
      - name: redisPassword
        secretKeyRef:
          name: redis
          key: redis-password
      - name: actorStateStore
        value: "true"
      - name: keyPrefix
        value: none
    EOF

  depends_on = [helm_release.dapr, data.kubernetes_service.redis]
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
      features:
        - name: Resiliency
          enabled: true
    EOF

  depends_on = [helm_release.dapr]
}
