output "ready_to_deploy" {
  value = true

  depends_on = [
    kubernetes_persistent_volume_claim.user_storage_pvc,
    helm_release.redis,
    helm_release.rabbitmq,
    kubectl_manifest.control-pubsub-sidecar,
    kubectl_manifest.resiliency-sidecar
  ]
}

output "shared_resource_pv_claim_name" {
  value = kubernetes_persistent_volume_claim.user_storage_pvc.metadata[0].name

  depends_on = [
    kubernetes_persistent_volume_claim.user_storage_pvc
  ]
}