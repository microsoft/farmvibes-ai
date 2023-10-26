How to use this terraform file?

Install Terraform from https://developer.hashicorp.com/terraform/downloads

In current directory, execute:

terraform init
terraform apply -var-file=example-vars.tfvars

Terraform apply will ask you the following questions:
You can also refer to example-vars.tfvars

location - This is the Azure Region you want to deploy in. For example, westus2, eastus2, etc.
tenantId - This is the Azure Tenant GUID of your Tenant. You can find this by going to Azure Active Directory or navigating to: https://ms.portal.azure.com/#view/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade/~/Overview
subscriptionId - This is the Subscription GUID for the subscription you want to us.
namespace - This is the kubernetes namespace you want to deploy your services in. This will be a new namespace which the script will create. Recommneded value is "terravibes"
acr_registry - This is the path to the Docker Registry where the images are location. Public location for FarmVibes is mcr.microsoft.com/farmai/terravibes
acr_registry_username - Username to access the Docker Registry
acr_registry_password - Password to access the Docker Registry
prefix - A short prefix to distinguish your deployment
resource_group_name - If you want to use an existing resource group, specify it here