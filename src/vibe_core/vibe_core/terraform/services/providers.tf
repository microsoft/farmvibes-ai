# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

terraform {
  required_version = ">=1.6.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "3.2.1"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "1.19.0"
    }
  }

  backend "azurerm" {
    container_name = "terraform-state"
    key            = "services.tfstate"
  }

}

provider "kubernetes" {
  config_path    = var.kubernetes_config_path
  config_context = var.kubernetes_config_context
}

provider "kubectl" {
  config_path      = var.kubernetes_config_path
  config_context   = var.kubernetes_config_context
  load_config_file = true
}
