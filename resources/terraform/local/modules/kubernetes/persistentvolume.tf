resource "kubernetes_persistent_volume" "user_storage_pv" {
  metadata {
    name = "user-storage-pv"
  }

  spec {
    access_modes = ["ReadWriteMany"]
    capacity = {
      storage = "2Gi"
    }
    persistent_volume_source {
      host_path {
        path = var.host_storage_path
        type = "Directory"
      }
    }

    storage_class_name = "manual"
    persistent_volume_reclaim_policy = "Retain"
  }
}

resource "kubernetes_persistent_volume_claim" "user_storage_pvc" {
  metadata {
    name = "user-storage-pvc"
    namespace = var.namespace
  }

  spec {
    access_modes = ["ReadWriteMany"]
    storage_class_name = "manual"
    resources {
      requests = {
        storage = "2Gi"
      }
    }
  }

  depends_on = [
    kubernetes_persistent_volume.user_storage_pv
  ]
}