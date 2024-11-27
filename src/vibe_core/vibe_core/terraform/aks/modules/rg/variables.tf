# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

variable "environment" {
  description = "Azure Cloud Environment to use"
}
