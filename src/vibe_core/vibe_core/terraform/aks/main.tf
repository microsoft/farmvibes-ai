terraform {
  required_version = ">=0.12"
}

module "rg" {
  source = "./modules/rg"
}

module "infrastructure" {
  source              = "./modules/infra"
  location            = var.location
  prefix              = var.prefix
  tenantId            = var.tenantId
  subscriptionId      = var.subscriptionId
  resource_group_name = var.resource_group_name
  max_worker_nodes    = var.worker_replicas
  farmvibes_log_level = var.farmvibes_log_level
  depends_on          = [module.rg]
}

module "kubernetes" {
  source                    = "./modules/kubernetes"
  tenantId                  = var.tenantId
  namespace                 = var.namespace
  acr_registry              = var.acr_registry
  acr_registry_username     = var.acr_registry_username
  acr_registry_password     = var.acr_registry_password
  kubernetes_config_path    = module.infrastructure.kubernetes_config_path
  kubernetes_config_context = module.infrastructure.kubernetes_config_context
  public_ip_address         = module.infrastructure.public_ip_address
  public_ip_fqdn            = module.infrastructure.public_ip_fqdn
  public_ip_dns             = module.infrastructure.public_ip_dns
  keyvault_name             = module.infrastructure.keyvault_name
  application_id            = module.infrastructure.application_id
  storage_connection_key    = module.infrastructure.storage_connection_key
  storage_account_name      = module.infrastructure.storage_account_name
  userfile_container_name   = module.infrastructure.userfile_container_name
  resource_group_name       = module.infrastructure.resource_group_name
  size_of_shared_volume     = var.size_of_shared_volume
  certificate_email         = var.certificate_email
  current_user_name         = module.infrastructure.current_user_name
}

module "services" {
  source                        = "../services"
  namespace                     = var.namespace
  prefix                        = var.prefix
  acr_registry                  = var.acr_registry
  kubernetes_config_path        = module.infrastructure.kubernetes_config_path
  kubernetes_config_context     = module.infrastructure.kubernetes_config_context
  worker_node_pool_name         = module.infrastructure.worker_node_pool_name
  public_ip_fqdn                = module.infrastructure.public_ip_fqdn
  dapr_sidecars_deployed        = module.kubernetes.dapr_sidecars_deployed
  startup_type                  = "aks"
  shared_resource_pv_claim_name = module.kubernetes.shared_resource_pv_claim_name
  image_prefix                  = var.image_prefix
  image_tag                     = var.image_tag
  worker_replicas               = var.worker_replicas
  farmvibes_log_level           = var.farmvibes_log_level
}
