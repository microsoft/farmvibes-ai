# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path


def test_remote_api_auth_kubernetes_contract():
    terraform = (
        Path(__file__).parents[1]
        / "vibe_core"
        / "terraform"
        / "services"
        / "restapi.tf"
    ).read_text()

    assert '"LoadBalancer"' not in terraform
    assert 'type = "ClusterIP"' in terraform
    assert "nginx.ingress.kubernetes.io/rewrite-target" not in terraform
    assert "nginx.ingress.kubernetes.io/use-regex" not in terraform
    assert (
        'dynamic "env" {\n'
        "            for_each = var.local_deployment ? [] : [1]\n"
        "            content {\n"
        '              name = "FARMVIBES_API_TOKEN"\n'
        "              value_from {\n"
        "                secret_key_ref {\n"
        '                  name = "farmvibes-api-auth"\n'
        '                  key  = "token"'
    ) in terraform

    assert (
        'ingress_class_name = var.local_deployment ? "traefik" : "traefik-remote"'
        in terraform
    )
    assert "farmvibes-https-redirect" in terraform
    assert "traefik.ingress.kubernetes.io/router.middlewares" in terraform
    assert "acme.cert-manager.io/http01-edit-in-place" not in terraform
    assert (
        'dynamic "tls" {\n'
        "      for_each = var.local_deployment ? [] : [1]\n"
        "      content {\n"
        "        hosts       = [var.public_ip_fqdn]\n"
        '        secret_name = "terravibes-rest-api-tls"'
    ) in terraform
    assert '"cert-manager.io/cluster-issuer" = "letsencrypt"' in terraform


def test_remote_services_use_pinned_native_images():
    root = Path(__file__).parents[1] / "vibe_core"
    terraform = root / "terraform"
    wrappers = (root / "cli" / "wrappers.py").read_text()
    redis = (terraform / "aks" / "modules" / "kubernetes" / "redis.tf").read_text()
    rabbitmq = (
        terraform / "aks" / "modules" / "kubernetes" / "rabbitmq.tf"
    ).read_text()

    assert '"redis_image": REDIS_IMAGE' in wrappers
    assert '"rabbitmq_image": RABBITMQ_IMAGE' in wrappers
    assert 'resource "helm_release"' not in redis + rabbitmq
    assert "image             = var.redis_image" in redis
    assert "image             = var.rabbitmq_image" in rabbitmq
    assert 'name = "rabbitmq-data"' in rabbitmq

    providers = (
        terraform / "aks" / "modules" / "kubernetes" / "providers.tf"
    ).read_text()
    assert 'source  = "hashicorp/random"' in providers
    services_providers = (terraform / "services" / "providers.tf").read_text()
    assert 'provider "helm"' not in services_providers
    assert 'backend "azurerm"' in services_providers

    ensure_services = wrappers.split("def ensure_services(", 1)[1].split(
        "def ensure_local_cluster(", 1
    )[0]
    assert "backend_config=backend_config" in ensure_services


def test_remote_aks_keeps_required_oidc_issuer_enabled():
    terraform = (
        Path(__file__).parents[1]
        / "vibe_core"
        / "terraform"
        / "aks"
        / "modules"
        / "infra"
        / "kubernetes.tf"
    ).read_text()

    assert "oidc_issuer_enabled       = true" in terraform


def test_cache_restart_keeps_one_replica_available():
    terraform = (
        Path(__file__).parents[1]
        / "vibe_core"
        / "terraform"
        / "services"
        / "cache.tf"
    ).read_text()

    assert "max_surge       = 1" in terraform
    assert "max_unavailable = 0" in terraform


def test_maintenance_infrastructure_versions_and_platforms():
    root = Path(__file__).parents[1] / "vibe_core"
    terraform = root / "terraform"
    infra = terraform / "aks" / "modules" / "infra"
    kubernetes = terraform / "aks" / "modules" / "kubernetes"
    local = terraform / "local" / "modules" / "kubernetes"

    assert 'version = "5.2.0"' in (infra / "providers.tf").read_text()
    assert 'version = "3.2.1"' in (kubernetes / "providers.tf").read_text()
    assert 'version = "3.2.0"' in (kubernetes / "providers.tf").read_text()
    cluster = (infra / "kubernetes.tf").read_text()
    assert cluster.count("AzureLinux3") == 2
    assert cluster.count("temporary_name_for_rotation") == 2
    assert 'version    = "1.18.3"' in (kubernetes / "dapr.tf").read_text()
    assert 'version    = "1.18.3"' in (local / "dapr.tf").read_text()
    ingress = (kubernetes / "init.tf").read_text()
    assert "https://traefik.github.io/charts" in ingress
    assert 'version    = "41.3.0"' in ingress
    cert_manager = (kubernetes / "cert.tf").read_text()
    assert 'version    = "v1.21.1"' in cert_manager
    assert 'name = "cert-manager"' in cert_manager
    assert "namespace  = kubernetes_namespace.cert_manager" in cert_manager
    assert 'name  = "clusterResourceNamespace"' in cert_manager
    assert 'value = "kube-system"' in cert_manager
    assert "atomic     = true" in cert_manager
    assert "timeout    = 600" in cert_manager
    assert 'name  = "crds.enabled"' in cert_manager


def test_local_integration_token_is_read_only():
    workflow = (
        Path(__file__).parents[3] / ".github" / "workflows" / "lint-test.yml"
    ).read_text()
    job = workflow.split("local-integration-tests:", 1)[1]

    assert "permissions:\n      contents: read" in job
    assert "GITHUB_TOKEN: ${{ github.token }}" in job
