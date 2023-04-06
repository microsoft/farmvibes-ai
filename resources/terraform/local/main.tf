terraform {
  required_version = ">=0.12"
}

module "kubernetes" {
  source                                      = "./modules/kubernetes"
  namespace                                   = var.namespace
  kubernetes_config_path                      = var.kubernetes_config_path
  kubernetes_config_context                   = var.kubernetes_config_context
  host_storage_path                           = var.host_storage_path
  redis_image_tag                             = var.redis_image_tag
  rabbitmq_image_tag                          = var.rabbitmq_image_tag
}

module "services" {
  source                                      = "../services"
  namespace                                   = var.namespace
  prefix                                      = "local"
  acr_registry                                = var.acr_registry
  run_as_user_id                              = var.run_as_user_id
  run_as_group_id                             = var.run_as_group_id
  working_dir                                 = "/tmp"
  log_dir                                     = "/mnt/logs"
  farmvibes_log_level                         = var.farmvibes_log_level
  host_assets_dir                             = var.host_assets_dir
  kubernetes_config_path                      = var.kubernetes_config_path
  kubernetes_config_context                   = var.kubernetes_config_context
  cache_node_pool_name                        = var.node_pool_name
  worker_node_pool_name                       = var.node_pool_name
  default_node_pool_name                      = var.node_pool_name
  public_ip_fqdn                              = ""
  dapr_sidecars_deployed                      = module.kubernetes.ready_to_deploy
  image_tag                                   = var.image_tag
  local_deployment                            = true
  worker_memory_request                       = "100Mi"
  startup_type                                = "local"
  shared_resource_pv_claim_name               = module.kubernetes.shared_resource_pv_claim_name
  worker_replicas                             = var.worker_replicas
  image_prefix                                = var.image_prefix
}
