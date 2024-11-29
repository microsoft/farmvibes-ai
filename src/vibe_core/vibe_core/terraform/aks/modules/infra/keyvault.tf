# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

data "http" "ip" {
  url = "https://ipv4.icanhazip.com"
}

resource "azurerm_key_vault" "keyvault" {
  name                          = "${var.prefix}-kv-${resource.random_string.name_suffix.result}"
  location                      = var.location
  resource_group_name           = var.resource_group_name
  enabled_for_disk_encryption   = true
  tenant_id                     = var.tenantId
  soft_delete_retention_days    = 7
  purge_protection_enabled      = false
  public_network_access_enabled = true
  sku_name                      = "standard"

  access_policy {
    tenant_id = var.tenantId
    object_id = data.azurerm_user_assigned_identity.kubernetesidentity.principal_id

    key_permissions = [
      "Get", "List",
    ]

    secret_permissions = [
      "Get", "List",
    ]
  }

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Create",
      "Get",
    ]

   secret_permissions = [
       "Get",  "Backup", "Delete", "List", "Purge", "Recover", "Restore", "Set"
  ]
}

  network_acls {
    bypass                     = "AzureServices"
    default_action             = "Allow"
    virtual_network_subnet_ids = [azurerm_subnet.aks-subnet.id]
    ip_rules                   = [trimspace(data.http.ip.response_body)]
  }

  depends_on = [data.azurerm_resource_group.resourcegroup, data.http.ip, data.azurerm_user_assigned_identity.kubernetesidentity]
}

resource "azurerm_key_vault_secret" "cosmosdbsecret" {
  name         = "cosmos-db-database"
  value        = azurerm_cosmosdb_sql_database.cosmosdb.name
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_sql_database.cosmosdb]
}

resource "azurerm_key_vault_secret" "cosmoscollectionsecret" {
  name         = "cosmos-db-collection"
  value        = azurerm_cosmosdb_sql_container.workflows.name
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_sql_container.workflows]
}

resource "azurerm_key_vault_secret" "cosmosdbkey" {
  name         = "cosmos-db-key"
  value        = azurerm_cosmosdb_account.cosmos.primary_key
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_account.cosmos]
}

resource "azurerm_key_vault_secret" "cosmosdburl" {
  name         = "cosmos-db-url"
  value        = azurerm_cosmosdb_account.cosmos.endpoint
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_account.cosmos]
}

resource "azurerm_key_vault_secret" "storageconnectionstring" {
  name         = "storage-account-connection-string"
  value        = azurerm_storage_account.storageaccount.primary_connection_string
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_storage_account.storageaccount]
}

resource "azurerm_key_vault_secret" "staccosmosuri" {
  name         = "stac-cosmos-db-url"
  value        = azurerm_cosmosdb_account.staccosmos.endpoint
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_account.staccosmos]
}

resource "azurerm_key_vault_secret" "staccosmoskeysecret" {
  name         = "stac-cosmos-write-key"
  value        = azurerm_cosmosdb_account.staccosmos.primary_key
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_account.staccosmos]
}

resource "azurerm_key_vault_secret" "staccosmosdbname" {
  name         = "stac-cosmos-db-name"
  value        = azurerm_cosmosdb_sql_database.cosmosstacdb.name
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_sql_database.cosmosstacdb]
}

resource "azurerm_key_vault_secret" "staccosmoscontainer" {
  name         = "stac-cosmos-container-name"
  value        = azurerm_cosmosdb_sql_container.staccontainer.name
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_sql_container.staccontainer]
}

resource "azurerm_key_vault_secret" "staccosmosassetscontainer" {
  name         = "stac-cosmos-assets-container-name"
  value        = azurerm_cosmosdb_sql_container.stacassetscontainer.name
  key_vault_id = azurerm_key_vault.keyvault.id
  depends_on   = [azurerm_key_vault.keyvault, azurerm_cosmosdb_sql_container.stacassetscontainer]
}
