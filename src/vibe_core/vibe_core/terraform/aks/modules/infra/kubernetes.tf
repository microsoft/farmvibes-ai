# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

locals {
  default_node_pool_max_count = 3
}

resource "azurerm_kubernetes_cluster" "kubernetes" {
  name                      = var.prefix
  location                  = var.location
  resource_group_name       = var.resource_group_name
  dns_prefix                = "${var.prefix}kbsdns"
  automatic_upgrade_channel = "patch"
  oidc_issuer_enabled       = true

  identity {
    type = "SystemAssigned"
  }

  node_provisioning_profile {
    mode = "Manual"
  }

  role_based_access_control_enabled = true

  azure_active_directory_role_based_access_control {
    azure_rbac_enabled = true
    tenant_id          = var.tenantId
  }

  default_node_pool {
    name                        = "default"
    auto_scaling_enabled        = true
    min_count                   = 2
    max_count                   = local.default_node_pool_max_count
    vm_size                     = "Standard_B4ms"
    os_sku                      = "AzureLinux3"
    vnet_subnet_id              = azurerm_subnet.aks-subnet.id
    temporary_name_for_rotation = "defaulttmp"
  }

  storage_profile {
    blob_driver_enabled = true
  }

  network_profile {
    network_plugin = "azure"
  }

  depends_on = [azurerm_subnet.aks-subnet, data.azurerm_resource_group.resourcegroup]
}

data "azurerm_user_assigned_identity" "kubernetesidentity" {
  name                = "${azurerm_kubernetes_cluster.kubernetes.name}-agentpool"
  resource_group_name = azurerm_kubernetes_cluster.kubernetes.node_resource_group

  depends_on = [azurerm_kubernetes_cluster.kubernetes]
}


resource "azurerm_kubernetes_cluster_node_pool" "kubernetes-worker" {
  name                        = "worker"
  kubernetes_cluster_id       = azurerm_kubernetes_cluster.kubernetes.id
  vm_size                     = "Standard_D8s_v3"
  auto_scaling_enabled        = true
  min_count                   = 1
  max_count                   = var.max_worker_nodes
  os_sku                      = "AzureLinux3"
  temporary_name_for_rotation = "workertmp"
  depends_on                  = [azurerm_kubernetes_cluster.kubernetes]

  lifecycle {
    ignore_changes = [
      vnet_subnet_id,
    ]
  }
}

resource "local_file" "kubeconfig" {
  filename        = "${var.kubeconfig_location}/kubeconfig"
  content         = azurerm_kubernetes_cluster.kubernetes.kube_admin_config_raw
  file_permission = "0600"
  depends_on      = [azurerm_kubernetes_cluster.kubernetes]
}
