variable "tenantId" {
  description = "Tenant ID"
}

variable "namespace" {
  description = "Namespace"
}

variable "acr_registry" {
  description = "ACR Registry"
}

variable "acr_registry_username" {
  description = "ACR Registry Username"
}

variable "acr_registry_password" {
  description = "ACR Registry Password"
}

variable "kubernetes_config_path" {
  description = "Path where kubeconfig is located"
}

variable "kubernetes_config_context" {
}

variable "public_ip_address" {
}

variable "public_ip_fqdn" {
}

variable "public_ip_dns" {
}

variable "keyvault_name" {
}

variable "application_id" {
}

variable "storage_connection_key" {
}

variable "storage_account_name" {
}

variable "userfile_container_name" {
}

variable "resource_group_name" {
}

variable "size_of_shared_volume" {
  default = "10Gi"
}

variable "certificate_email" {
  description = "Email to send information about certificates being generated"
}

variable "current_user_name" {
  description = "Current user name, used to add to the cluster-admin role"
}