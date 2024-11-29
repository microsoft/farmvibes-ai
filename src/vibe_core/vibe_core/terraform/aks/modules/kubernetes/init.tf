resource "kubernetes_namespace" "kubernetesnamespace" {
  count = var.namespace == "default" ? 0 : 1
  metadata {
    name = var.namespace
  }
}

data "kubernetes_namespace" "kubernetesnamespace" {
  metadata {
    name = var.namespace == "default" ? var.namespace : kubernetes_namespace.kubernetesnamespace[0].metadata.0.name
  }
}

resource "kubernetes_secret" "user-storage-secret" {
  metadata {
    name      = "user-storage-secret"
    namespace = var.namespace
  }

  data = {
    "azurestorageaccountname" = var.storage_account_name,
    "azurestorageaccountkey"  = var.storage_connection_key
  }

  type = "Opaque"

  depends_on = [data.kubernetes_namespace.kubernetesnamespace]
}

resource "kubernetes_secret" "monitor_instrumentation_key_secret" {
  metadata {
    name = "monitor-instrumentation-key-secret"
    namespace = var.namespace
  }

  data = {
    monitor_instrumentation_key = var.monitor_instrumentation_key
  }

  type = "Opaque"
  depends_on = [data.kubernetes_namespace.kubernetesnamespace]
}

resource "kubernetes_secret" "eywaregistrysecret" {
  metadata {
    name      = "acrtoken"
    namespace = var.namespace
  }

  type = "kubernetes.io/dockerconfigjson"

  data = {
    ".dockerconfigjson" = jsonencode({
      auths = {
        "${var.acr_registry}" = {
          "username" = var.acr_registry_username
          "password" = var.acr_registry_password
          "email"    = var.certificate_email
          "auth"     = base64encode("${var.acr_registry_username}:${var.acr_registry_password}")
        }
      }
    })
  }

  depends_on = [data.kubernetes_namespace.kubernetesnamespace]
}

resource "kubernetes_namespace" "kubernetesnginxnamespace" {
  metadata {
    name = "ingress-basic"
  }
}

resource "helm_release" "nginx-ingress" {
  name       = "ingress-nginx"
  repository = "https://helm.nginx.com/stable"
  chart      = "nginx-ingress"
  namespace  = "ingress-basic"
  timeout    = 600
  version    = "0.16.0"

  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/azure-load-balancer-health-probe-request-path"
    value = "/healthz"
  }
  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/azure-dns-label-name"
    value = var.public_ip_dns
  }
  set {
    name  = "controller.service.loadBalancerIP"
    value = var.public_ip_address
  }
  depends_on = [kubernetes_namespace.kubernetesnginxnamespace]
}

resource "kubernetes_cluster_role_binding" "admin" {
  metadata {
    name = "az-cli-admin"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = "cluster-admin"
  }

  subject {
    kind      = "User"
    name      = var.current_user_name
    api_group = "rbac.authorization.k8s.io"
  }

  depends_on = [data.kubernetes_namespace.kubernetesnamespace]
}