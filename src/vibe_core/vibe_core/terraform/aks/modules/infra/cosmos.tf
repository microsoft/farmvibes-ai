resource "azurerm_cosmosdb_account" "cosmos" {
  name                              = "${var.prefix}-cosmos-${resource.random_string.name_suffix.result}"
  location                          = var.location
  resource_group_name               = var.resource_group_name
  offer_type                        = "Standard"
  kind                              = "GlobalDocumentDB"
  enable_automatic_failover         = false
  public_network_access_enabled     = true
  is_virtual_network_filter_enabled = true

  geo_location {
    location          = var.location
    failover_priority = 0
  }

  capabilities {
    name = "EnableServerless"
  }

  consistency_policy {
    consistency_level       = "BoundedStaleness"
    max_interval_in_seconds = 300
    max_staleness_prefix    = 100000
  }

  virtual_network_rule {
    id                                   = azurerm_subnet.aks-subnet.id
    ignore_missing_vnet_service_endpoint = false
  }
  depends_on = [
    data.azurerm_resource_group.resourcegroup, azurerm_subnet.aks-subnet
  ]
}

resource "azurerm_cosmosdb_sql_database" "cosmosdb" {
  name                = "database"
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.cosmos.name
}

resource "azurerm_cosmosdb_sql_container" "workflows" {
  name                  = "workflows"
  resource_group_name   = var.resource_group_name
  account_name          = azurerm_cosmosdb_account.cosmos.name
  database_name         = azurerm_cosmosdb_sql_database.cosmosdb.name
  partition_key_path    = "/partitionKey"
  partition_key_version = 1

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }
  }
}

resource "azurerm_cosmosdb_account" "staccosmos" {
  name                              = "${var.prefix}-cosmos-stac-${resource.random_string.name_suffix.result}"
  location                          = var.location
  resource_group_name               = var.resource_group_name
  offer_type                        = "Standard"
  kind                              = "GlobalDocumentDB"
  enable_automatic_failover         = false
  public_network_access_enabled     = true
  is_virtual_network_filter_enabled = true

  geo_location {
    location          = var.location
    failover_priority = 0
  }

  consistency_policy {
    consistency_level       = "BoundedStaleness"
    max_interval_in_seconds = 300
    max_staleness_prefix    = 100000
  }

  virtual_network_rule {
    id                                   = azurerm_subnet.aks-subnet.id
    ignore_missing_vnet_service_endpoint = false
  }
  depends_on = [
    data.azurerm_resource_group.resourcegroup, azurerm_subnet.aks-subnet
  ]
}

resource "azurerm_cosmosdb_sql_database" "cosmosstacdb" {
  name                = "stacdb"
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.staccosmos.name
}

resource "azurerm_cosmosdb_sql_container" "staccontainer" {
  name                  = "stac"
  resource_group_name   = var.resource_group_name
  account_name          = azurerm_cosmosdb_account.staccosmos.name
  database_name         = azurerm_cosmosdb_sql_database.cosmosstacdb.name
  partition_key_path    = "/op_name"
  partition_key_version = 1

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }
  }
}

resource "azurerm_cosmosdb_sql_container" "stacassetscontainer" {
  name                  = "stacassets"
  resource_group_name   = var.resource_group_name
  account_name          = azurerm_cosmosdb_account.staccosmos.name
  database_name         = azurerm_cosmosdb_sql_database.cosmosstacdb.name
  partition_key_path    = "/op_name"
  partition_key_version = 1

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }
  }
}
