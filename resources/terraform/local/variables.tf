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
}

variable "host_storage_path" {
}

variable "worker_replicas" {
  default = 1
}

variable "image_prefix" {
}
