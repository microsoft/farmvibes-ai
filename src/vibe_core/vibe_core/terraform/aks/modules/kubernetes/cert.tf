# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "helm_release" "letsencrypt" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "kube-system"
  version    = "1.12.2"

  set {
    name  = "installCRDs"
    value = "true"
  }

  set {
    name  = "nodeSelector.kubernetes\\.io/os"
    value = "linux"
  }

  depends_on = [helm_release.nginx-ingress]
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
              class: nginx
              podTemplate:
                spec:
                  nodeSelector:
                    "kubernetes.io/os": linux
    EOF

  depends_on = [helm_release.letsencrypt]
}