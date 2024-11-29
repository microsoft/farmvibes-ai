resource "helm_release" "redis" {
  name = "redis"

  repository = "https://charts.bitnami.com/bitnami"
  chart      = "redis"
  namespace  = var.namespace

  set {
    name  = "auth.enabled"
    value = "true"
  }

  set {
    name  = "master.containerPort"
    value = "6379"
  }

  set {
    name  = "replica.replicaCount"
    value = "0"
  }

  depends_on = [data.kubernetes_namespace.kubernetesnamespace]
}

data "kubernetes_service" "redis" {
  metadata {
    name      = "redis-master"
    namespace = var.namespace
  }

  depends_on = [helm_release.redis]
}
