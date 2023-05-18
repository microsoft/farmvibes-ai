# Terraform constants
DEFAULT_IMAGE_PREFIX = ""
DEFAULT_IMAGE_TAG = "2023.02.24"
DEFAULT_REGISTRY_PATH = "mcr.microsoft.com/farmai/terravibes"
REMOTE_SERVICE_URL_PATH_FILE = "remote_service_url"
MAX_WORKER_NODES = 2
PREFIX_LENGTH = 6
PREFIX_FILE_NAME = "remote_prefix"
TERRAFORM_FOLDER_NAME = "terraform_state"

# Azure constants
AZ_CREDS_REFRESH_ATTEMPTS = 2
TOTAL_REGIONAL_CPU_NAME = "Total Regional vCPUs"
WORKER_NODE_CPU_NAME = "Standard Av2 Family vCPUs"
CACHE_NODE_CPU_NAME = "Standard BS Family vCPUs"
AZ_LOGIN_PROMPT = "`az login`"

# Logging constants
LOGGING_LEVEL_INFO = 1
LOGGING_LEVEL_VERBOSE = 2

# Internal Tools Constants
FRIENDLY_NAME = "friendly_name"
MIN_VERSION_REQUIRED_NAME = "min_version"
AUTO_INSTALL_AVAILABLE_NAME = "can_auto_install"
INSTALL_ACTION_NAME = "install_action"
UPGRADE_ACTION_NAME = "upgrade_action"
VERSION_SELECTOR_CMD_NAME = "version_selector"
ALTERNATE_NAMES = "alternate_names"

TERRAFORM_RELEASE_PATH = "https://api.github.com/repos/hashicorp/terraform/releases/latest"
KUBECTL_BASE_PATH = "https://storage.googleapis.com/kubernetes-release/release"
HELM_RELEASE_PATH = "https://github.com/helm/helm/releases"
HELM_VERSION_SEARCH_STRING = 'href="/helm/helm/releases/tag/v'

REQUIRED_TOOLS = {
    "terraform": {
        FRIENDLY_NAME: "HashiCorp Terraform",
        MIN_VERSION_REQUIRED_NAME: "1.0.2",
    },
    "az": {
        FRIENDLY_NAME: "Microsoft Azure CLI",
        MIN_VERSION_REQUIRED_NAME: "2.46.0",
    },
    "helm": {
        FRIENDLY_NAME: "Helm",
        MIN_VERSION_REQUIRED_NAME: "1.0.0",
    },
    "kubectl": {
        FRIENDLY_NAME: "Kubectl",
        MIN_VERSION_REQUIRED_NAME: "1.0.0",
    },
}
