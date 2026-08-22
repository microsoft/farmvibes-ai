# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

variable "namespace" {
  description = "Namespace"
}

variable "kubernetes_config_path" {
}

variable "kubernetes_config_context" {
}

variable "host_storage_path" {
}

variable "redis_image" {
}

variable "rabbitmq_image" {
}

variable "enable_telemetry" {
  description = "Use telemetry"
  type        = bool
}