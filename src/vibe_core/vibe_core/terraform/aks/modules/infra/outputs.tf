# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

output "kubernetes_config_path" {
  value = local_file.kubeconfig.filename

  depends_on = [
    local_file.kubeconfig,
    azurerm_kubernetes_cluster.kubernetes
  ]
}

output "kubernetes_config_context" {
  value      = azurerm_kubernetes_cluster.kubernetes.name
  depends_on = [azurerm_kubernetes_cluster.kubernetes]
}

output "worker_node_pool_name" {
  value = azurerm_kubernetes_cluster_node_pool.kubernetes-worker.name
}

output "public_ip_address" {
  value = azurerm_public_ip.publicip.ip_address
}

output "public_ip_fqdn" {
  value = azurerm_public_ip.publicip.fqdn
}

output "public_ip_dns" {
  value = "${var.prefix}-${substr(sha256(var.resource_group_name), 0, 6)}-dns"
}

output "keyvault_name" {
  value = azurerm_key_vault.keyvault.name
}

output "application_id" {
  value = data.azurerm_user_assigned_identity.kubernetesidentity.client_id
}

output "storage_account_name" {
  value = azurerm_storage_account.storageaccount.name
}

output "userfile_container_name" {
  value = azurerm_storage_container.userfiles.name
}

output "storage_connection_key" {
  value     = azurerm_storage_account.storageaccount.primary_access_key
  sensitive = true
}

output "max_worker_nodes" {
  value     = azurerm_kubernetes_cluster_node_pool.kubernetes-worker.max_count
}

output "max_default_nodes" {
  value     = azurerm_kubernetes_cluster.kubernetes.default_node_pool[0].max_count
}

output "monitor_instrumentation_key" {
  value     = var.enable_telemetry ? azurerm_application_insights.appinsights[0].instrumentation_key : ""
  sensitive = true
}