variable "prefix" {
  description = "Prefix for resources"
}

variable "namespace" {
  description = "Namespace"
}

variable "kubernetes_config_path" {
}

variable "kubernetes_config_context" {
}

variable "worker_node_pool_name" {
}

variable "default_node_pool_name" {
  default = "default"
}

variable "acr_registry" {
}

variable "public_ip_fqdn" {
}

variable "dapr_sidecars_deployed" {
}

variable "working_dir" {
  default = ""
}

variable "run_as_user_id" {
  default = ""
}

variable "run_as_group_id" {
  default = ""
}

variable "log_dir" {
  default = ""
}

variable "max_log_file_bytes" {
  default = ""
}

variable "log_backup_count" {
  default = ""
}

variable "host_assets_dir" {
  default = ""
}

variable "local_deployment" {
  default = false
}

variable "image_prefix" {
  default = "terravibes-"
}

variable "image_tag" {
  default = "latest"
}

variable "worker_memory_request" {
  default = "8Gi"
}

variable "startup_type" {
}

variable "shared_resource_pv_claim_name" {
}

variable "otel_service_name" {
}

variable "worker_replicas" {
  default = 1
}

variable "farmvibes_log_level" {
  default = "INFO"
}

variable "environment" {
  description = "Unused"
  default = ""
}
