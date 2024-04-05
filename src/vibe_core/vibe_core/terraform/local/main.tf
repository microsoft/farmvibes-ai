terraform {
  required_version = ">=0.12"
  backend "local" {
    path = "~/.config/farmvibes-ai/local.tfstate"
  }
}

module "kubernetes" {
  source                    = "./modules/kubernetes"
  namespace                 = var.namespace
  kubernetes_config_path    = var.kubernetes_config_path
  kubernetes_config_context = var.kubernetes_config_context
  host_storage_path         = var.host_storage_path
  redis_image_tag           = var.redis_image_tag
  rabbitmq_image_tag        = var.rabbitmq_image_tag
  enable_telemetry          = var.enable_telemetry
}

module "services" {
  source                        = "../services"
  namespace                     = var.namespace
  prefix                        = "local"
  acr_registry                  = var.acr_registry
  run_as_user_id                = var.run_as_user_id
  run_as_group_id               = var.run_as_group_id
  working_dir                   = "/tmp"
  log_dir                       = "/mnt/logs"
  farmvibes_log_level           = var.farmvibes_log_level
  max_log_file_bytes            = var.max_log_file_bytes
  log_backup_count              = var.log_backup_count
  host_assets_dir               = var.host_assets_dir
  kubernetes_config_path        = var.kubernetes_config_path
  kubernetes_config_context     = var.kubernetes_config_context
  worker_node_pool_name         = var.node_pool_name
  default_node_pool_name        = var.node_pool_name
  public_ip_fqdn                = ""
  dapr_sidecars_deployed        = module.kubernetes.ready_to_deploy
  image_tag                     = var.image_tag
  local_deployment              = true
  worker_memory_request         = "100Mi"
  startup_type                  = "local"
  shared_resource_pv_claim_name = module.kubernetes.shared_resource_pv_claim_name
  otel_service_name             = try(module.kubernetes.otel_service_name, "")
  worker_replicas               = var.worker_replicas
  image_prefix                  = var.image_prefix
}
