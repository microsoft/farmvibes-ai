variable "location" {
  description = "Location of the resources."
}

variable "prefix" {
  description = "Prefix of the resources."
}

variable "tenantId" {
  description = "Tenant ID."
}

variable "subscriptionId" {
  description = "Subscription ID"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  default     = null
}

variable "kubeconfig_location" {
  description = "Location where to store kubeconfig file for the AKS cluster created"
}

variable "max_worker_nodes" {
  description = "Maximum number of nodes for a worker"
}

variable "environment" {
  description = "Azure Cloud Environment to use"
}
