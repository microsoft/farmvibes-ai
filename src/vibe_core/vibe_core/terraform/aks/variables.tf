variable "location" {
  description = "Azure Location of the resources."
}

variable "prefix" {
  description = "Prefix of the resources. (3-5 chars)"
}

variable "tenantId" {
  description = "Azure Tenant ID."
}

variable "subscriptionId" {
  description = "Subscription ID"
}

variable "namespace" {
  description = "Namespace"
}

variable "acr_registry" {
  description = "ACR Registry"
}

variable "acr_registry_username" {
  description = "ACR Registry Username"
  default     = ""
}

variable "acr_registry_password" {
  description = "ACR Registry Password"
  default     = ""
}

variable "resource_group_name" {
  description = "If you want use an existing RG, specify it here. Else leave empty. Should be in the same Location as requested"
  default     = null
}

variable "enable_telemetry" {
  description = "Use telemetry"
  type        = bool
}

variable "monitor_instrumentation_key" {
  description = "Instrumentation Key for Azure Monitor"
  default     = null
}

variable "image_prefix" {
  default = "terravibes-"
}

variable "image_tag" {
}

variable "worker_replicas" {
  default = 1
}

variable "size_of_shared_volume" {
  default = "10Gi"
}

variable "certificate_email" {
  description = "Email to send information about certificates being generated"
}

variable "farmvibes_log_level" {
  description = "Log level to use with FarmVibes.AI services"
}