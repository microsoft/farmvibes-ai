variable "acr_registry" {
  description = "ACR Registry"
}

variable "namespace" {
  default = "default"
}

variable "run_as_user_id" {
}

variable "run_as_group_id" {
}

variable "host_assets_dir" {
}

variable "kubernetes_config_path" {
  default = "~/.kube/config"
}

variable "kubernetes_config_context" {
}

variable "image_tag" {
}

variable "node_pool_name" {
  default = ""
}

variable "host_storage_path" {
}

variable "worker_replicas" {
  default = 1
}

variable "image_prefix" {
  default     = ""
  description = "Prefix for the image name"
}

variable "redis_image_tag" {
}

variable "rabbitmq_image_tag" {
}

variable "enable_telemetry" {
  description = "Use telemetry"
  type        = bool
}

variable "farmvibes_log_level" {
  default     = "INFO"
  description = "Log level to use with FarmVibes.AI services"
}

variable "max_log_file_bytes" {
 description = "Maximum size of a worker log file in bytes"
}

variable "log_backup_count" {
 description = "Number of log files to keep for each service instance"
}

variable "environment" {
  description = "Unused"
  default = ""
}
