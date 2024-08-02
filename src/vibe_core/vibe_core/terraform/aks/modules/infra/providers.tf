# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

  backend "azurerm" {
    container_name = "terraform-state"
    key            = "infra.tfstate"
  }
}

provider "azurerm" {
  tenant_id                  = var.tenantId
  subscription_id            = var.subscriptionId
  skip_provider_registration = "true"
  features {}
}

provider "random" {}