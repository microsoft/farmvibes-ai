resource "azurerm_public_ip" "publicip" {
  name                = "${var.prefix}-${substr(sha256(var.resource_group_name), 0, 6)}-ip"
  resource_group_name = azurerm_kubernetes_cluster.kubernetes.node_resource_group
  location            = var.location
  allocation_method   = "Static"
  sku                 = "Standard"
  domain_name_label   = "${var.prefix}-${substr(sha256(var.resource_group_name), 0, 6)}-dns"
  depends_on          = [azurerm_kubernetes_cluster.kubernetes]
}
