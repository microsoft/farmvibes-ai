resource "azurerm_log_analytics_workspace" "analyticsworkspace" {
  name                = "${var.prefix}-analytics-workspace-${resource.random_string.name_suffix.result}"
  count               = var.enable_telemetry ? 1 : 0
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
}

resource "azurerm_application_insights" "appinsights" {
  name                = "${var.prefix}-app-insights-${resource.random_string.name_suffix.result}"
  count               = var.enable_telemetry ? 1 : 0
  location            = var.location
  resource_group_name = var.resource_group_name
  application_type    = "web"
}


resource "azurerm_monitor_diagnostic_setting" "diagsetting" {
  name                       = "${var.prefix}-diagsetting-${resource.random_string.name_suffix.result}"
  count                      = var.enable_telemetry ? 1 : 0
  target_resource_id         = azurerm_application_insights.appinsights[0].id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.analyticsworkspace[0].id

  enabled_log {
    category = "AppTraces"

    retention_policy {
      enabled = false
    }
  }

  metric {
    category = "AllMetrics"

    retention_policy {
      enabled = false
    }
  }
}
