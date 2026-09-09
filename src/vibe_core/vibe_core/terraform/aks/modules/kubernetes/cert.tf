# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "kubernetes_namespace" "cert_manager" {
  metadata {
    name = "cert-manager"
  }
}

resource "helm_release" "letsencrypt" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = kubernetes_namespace.cert_manager.metadata[0].name
  version    = "v1.21.1"
  atomic     = true
  timeout    = 600

  set = [
    {
      name  = "crds.enabled"
      value = "true"
    },
    {
      name  = "nodeSelector.kubernetes\\.io/os"
      value = "linux"
    },
    {
      name  = "clusterResourceNamespace"
      value = "kube-system"
    }
  ]

  depends_on = [helm_release.nginx-ingress, kubernetes_namespace.cert_manager]
}

resource "kubectl_manifest" "clusterissuer" {
  yaml_body = <<-EOF
    apiVersion: cert-manager.io/v1
    kind: ClusterIssuer
    metadata:
      name: letsencrypt
      namespace: kube-system
    spec:
      acme:
        server: https://acme-v02.api.letsencrypt.org/directory
        email: ${var.certificate_email}
        privateKeySecretRef:
          name: letsencrypt
        solvers:
        - http01:
            ingress:
              class: traefik-remote
              podTemplate:
                spec:
                  nodeSelector:
                    "kubernetes.io/os": linux
    EOF

  depends_on = [helm_release.letsencrypt]
}