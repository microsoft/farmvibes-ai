resource "azurerm_storage_account" "storageaccount" {
  name                          = "storage${resource.random_string.name_suffix.result}"
  resource_group_name           = var.resource_group_name
  location                      = var.location
  account_tier                  = "Standard"
  account_replication_type      = "LRS"
  min_tls_version               = "TLS1_2"
  public_network_access_enabled = true
  network_rules {
    default_action             = "Allow"
    bypass                     = ["AzureServices"]
    virtual_network_subnet_ids = [azurerm_subnet.aks-subnet.id]
  }

  lifecycle {
    ignore_changes = [
      allow_nested_items_to_be_public,
      network_rules,
    ]
  }

}

resource "azurerm_storage_container" "userfiles" {
  name                  = "user-files"
  storage_account_name  = azurerm_storage_account.storageaccount.name
  container_access_type = "private"
}
