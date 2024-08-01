# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

data "azurerm_resource_group" "resourcegroup" {
  name = var.resource_group_name
}