# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "azurerm_network_security_group" "aks-nsg" {
  name                = "${var.prefix}-nsg"
  location            = var.location
  resource_group_name = var.resource_group_name

  security_rule {
    name                       = "allow_http"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "allow_https"
    priority                   = 1101
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  depends_on = [data.azurerm_resource_group.resourcegroup]
}

resource "azurerm_virtual_network" "vnet" {
  name                = "${var.prefix}-vnettf"
  location            = var.location
  resource_group_name = var.resource_group_name
  address_space       = ["10.224.0.0/12"]
  depends_on          = [data.azurerm_resource_group.resourcegroup]
}

resource "azurerm_subnet" "aks-subnet" {
  name                 = "aks"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.224.0.0/16"]
  service_endpoints    = ["Microsoft.AzureCosmosDB", "Microsoft.KeyVault", "Microsoft.ServiceBus", "Microsoft.Storage"]
  depends_on           = [data.azurerm_resource_group.resourcegroup, azurerm_virtual_network.vnet]
}

resource "azurerm_subnet_network_security_group_association" "aks-subnet-nsg" {
  subnet_id                 = azurerm_subnet.aks-subnet.id
  network_security_group_id = azurerm_network_security_group.aks-nsg.id
  depends_on                = [azurerm_subnet.aks-subnet, azurerm_network_security_group.aks-nsg]
}