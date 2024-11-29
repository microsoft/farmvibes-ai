terraform {
  required_version = ">=0.12"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">=2.16.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">=2.7.1"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.7.0"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubernetes_config_path
  config_context = var.kubernetes_config_context

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "kubelogin"
    args = [
      "get-token",
      "--use-azurerm-env-vars",
      "|",
      "jq",
      ".status.token"
    ]
  }
}

provider "helm" {
  kubernetes {
    config_path    = var.kubernetes_config_path
    config_context = var.kubernetes_config_context
  }
}

provider "kubectl" {
  config_path      = var.kubernetes_config_path
  config_context   = var.kubernetes_config_context
  load_config_file = true
}