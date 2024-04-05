output "dapr_sidecars_deployed" {
  value = true

  depends_on = [
    kubectl_manifest.keyvaultsidecar,
    kubectl_manifest.statestore-sidecar,
    kubectl_manifest.daprconfigcollector,
    kubectl_manifest.resiliency-sidecar,
    kubectl_manifest.control-pubsub-sidecar,
    kubernetes_persistent_volume_claim.user_storage_pvc
  ]
}

output "shared_resource_pv_claim_name" {
  value = kubernetes_persistent_volume_claim.user_storage_pvc.metadata[0].name

  depends_on = [
    kubernetes_persistent_volume_claim.user_storage_pvc
  ]
}