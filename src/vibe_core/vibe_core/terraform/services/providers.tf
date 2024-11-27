# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

  backend "kubernetes" {
    secret_suffix = "terraform-state"
  }
}

provider "kubernetes" {
  config_path    = var.kubernetes_config_path
  config_context = var.kubernetes_config_context
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
