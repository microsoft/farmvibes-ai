output "ready_to_deploy" {
  value = true

  depends_on = [
    kubernetes_persistent_volume_claim.user_storage_pvc
  ]
}

output "shared_resource_pv_claim_name" {
  value = kubernetes_persistent_volume_claim.user_storage_pvc.metadata[0].name

  depends_on = [
    kubernetes_persistent_volume_claim.user_storage_pvc
  ]
}