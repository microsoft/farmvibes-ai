# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

terraform {
  required_version = ">=1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "5.2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "3.9.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "2.9.0"
    }
    http = {
      source  = "hashicorp/http"
      version = "3.6.1"
    }
  }

  backend "azurerm" {
    container_name = "terraform-state"
    key            = "infra.tfstate"
  }
}

provider "azurerm" {
  tenant_id                       = var.tenantId
  subscription_id                 = var.subscriptionId
  resource_provider_registrations = "none"
  features {}
}

provider "random" {}