terraform {
  required_version = ">=0.12"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "3.89.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "3.1.0"
    }
  }
}

provider "azurerm" {
  tenant_id                  = var.tenantId
  subscription_id            = var.subscriptionId
  skip_provider_registration = "true"
  features {}
}

provider "random" {}
