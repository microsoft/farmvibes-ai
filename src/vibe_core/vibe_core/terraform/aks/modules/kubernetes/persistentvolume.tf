# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "kubernetes_storage_class" "csi_storage_class" {
  metadata {
    name = "csi-storage-class"
  }

  storage_provisioner = "blob.csi.azure.com"
  parameters = {
    resourceGroup  = var.resource_group_name
    storageAccount = var.storage_account_name
    containerName  = var.userfile_container_name
  }

  reclaim_policy      = "Retain"
  volume_binding_mode = "Immediate"

  depends_on = [data.kubernetes_namespace.kubernetesnamespace]
}

resource "kubernetes_persistent_volume" "user_storage_pv" {
  metadata {
    name = "user-storage-pv"
  }

  spec {
    access_modes = ["ReadWriteMany"]
    capacity = {
      storage = var.size_of_shared_volume
    }
    persistent_volume_source {
      csi {
        driver        = "blob.csi.azure.com"
        read_only     = false
        volume_handle = "unique-user-storage"
        node_stage_secret_ref {
          name      = kubernetes_secret.user-storage-secret.metadata[0].name
          namespace = var.namespace
        }
        volume_attributes = {
          containerName = var.userfile_container_name
        }
      }
    }

    storage_class_name               = kubernetes_storage_class.csi_storage_class.metadata[0].name
    persistent_volume_reclaim_policy = "Retain"
  }

  depends_on = [
    data.kubernetes_namespace.kubernetesnamespace,
    kubernetes_storage_class.csi_storage_class,
    kubernetes_secret.user-storage-secret
  ]
}

resource "kubernetes_persistent_volume_claim" "user_storage_pvc" {
  metadata {
    name      = "user-storage-pvc"
    namespace = var.namespace
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class.csi_storage_class.metadata[0].name
    volume_name        = kubernetes_persistent_volume.user_storage_pv.metadata[0].name
    resources {
      requests = {
        storage = var.size_of_shared_volume
      }
    }
  }

  depends_on = [
    data.kubernetes_namespace.kubernetesnamespace,
    kubernetes_persistent_volume.user_storage_pv
  ]
}