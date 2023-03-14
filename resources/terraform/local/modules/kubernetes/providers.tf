terraform {
  required_version = ">=0.12"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">=2.16.0"
    }
  }
}

provider "kubernetes" {
  config_path        = "${var.kubernetes_config_path}"
  config_context     = "${var.kubernetes_config_context}"
}
