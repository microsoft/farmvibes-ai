# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "azurerm_resource_group" "resourcegroup" {
  location = var.location
  name     = var.resource_group_name
}
