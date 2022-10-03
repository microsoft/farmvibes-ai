@description('Location of your VM')
param location string = resourceGroup().location

@description('VM hardware specification')
param vm_size string = 'Standard_D8s_v3'

@description('VM computer name')
param computer_name string = 'farmvibes-ai-vm${uniqueString(resourceGroup().id)}'

@description('Name of the Network Security Group')
param network_security_group_name string = 'SecGroupNet'

@description('User machine SSH public key')
param ssh_public_key string

@description('The name of you Virtual Machine.')
param vm_suffix_name string

@description('VM name')
param vm_name string = 'farmvibes-ai-vm-${vm_suffix_name}'

@description('Unique DNS Name for the Public IP used to access the Virtual Machine.')
param dns_label_prefix string = toLower('farmvibes-ai-${vm_suffix_name}-${uniqueString(resourceGroup().id)}')

@description('Name of the VNET')
param virtual_network_name string = '${vm_name}-vnet'

@description('Name of the subnet in the virtual network')
param subnet_name string = '${vm_name}-subnet'

@description('Username for the Virtual Machine.')
param admin_username string = 'azureuser'

@description('Encoded script (base64) for VM execution')
param encoded_script string

var linux_configuration = {
    disablePasswordAuthentication: true
    ssh: {
        publicKeys: [
            {
                path: '/home/${admin_username}/.ssh/authorized_keys'
                keyData: ssh_public_key
            }
        ]
    }
}

var public_IP_address_name = '${vm_name}PublicIP'
var network_interface_name = '${vm_name}NetInt'
var subnet_address_prefix = '10.1.0.0/24'
var address_prefix = '10.1.0.0/16'
var os_disk_type = 'Standard_LRS'

resource nsg 'Microsoft.Network/networkSecurityGroups@2021-05-01' = {
    name: network_security_group_name
    location: location
    properties: {
        securityRules: [
            {
                name: 'SSH'
                properties: {
                    priority: 1000
                    protocol: 'Tcp'
                    access: 'Allow'
                    direction: 'Inbound'
                    sourceAddressPrefix: '*'
                    sourcePortRange: '*'
                    destinationAddressPrefix: '*'
                    destinationPortRange: '22'
                }
            }
        ]
    }
}

resource vnet 'Microsoft.Network/virtualNetworks@2021-05-01' = {
    name: virtual_network_name
    location: location
    properties: {
        addressSpace: {
            addressPrefixes: [
                address_prefix
            ]
        }
    }
}

resource subnet 'Microsoft.Network/virtualNetworks/subnets@2021-05-01' = {
    parent: vnet
    name: subnet_name
    properties: {
        addressPrefix: subnet_address_prefix
        privateEndpointNetworkPolicies: 'Enabled'
        privateLinkServiceNetworkPolicies: 'Enabled'
    }
}

resource public_IP 'Microsoft.Network/publicIPAddresses@2021-05-01' = {
    name: public_IP_address_name
    location: location
    sku: {
        name: 'Basic'
    }
    properties: {
        publicIPAllocationMethod: 'Dynamic'
        publicIPAddressVersion: 'IPv4'
        dnsSettings: {
            domainNameLabel: dns_label_prefix
        }
        idleTimeoutInMinutes: 4
    }
}

resource nic 'Microsoft.Network/networkInterfaces@2020-11-01' = {
    name: network_interface_name
    location: location
    properties: {
        ipConfigurations: [
            {
                name: 'ipconfig1'
                properties: {
                    subnet: {
                        id: subnet.id
                    }
                    privateIPAllocationMethod: 'Dynamic'
                    publicIPAddress: {
                        id: public_IP.id
                    }
                }
            }
        ]
        networkSecurityGroup: {
            id: nsg.id
        }
    }
}

resource ubuntu_vm 'Microsoft.Compute/virtualMachines@2020-12-01' = {
    name: vm_name
    location: location
    properties: {
        hardwareProfile: {
            vmSize: vm_size
        }
        osProfile: {
            computerName: computer_name
            adminUsername: admin_username
            linuxConfiguration: linux_configuration
        }
        storageProfile: {
            osDisk: {
                createOption: 'FromImage'
                managedDisk: {
                    storageAccountType: os_disk_type
                }
                diskSizeGB: 2048
            }
            imageReference: {
                publisher: 'Canonical'
                offer: '0001-com-ubuntu-server-focal'
                sku: '20_04-lts-gen2'
                version: 'latest'
            }
        }
        networkProfile: {
            networkInterfaces: [
                {
                    id: nic.id
                }
            ]
        }
    }
}

resource linux_vm_extensions 'Microsoft.Compute/virtualMachines/extensions@2019-07-01' = {
    parent: ubuntu_vm
    name: 'farmvibes-ai_setup_script'
    location: location
    properties: {
        publisher: 'Microsoft.Azure.Extensions'
        type: 'CustomScript'
        typeHandlerVersion: '2.1'
        autoUpgradeMinorVersion: true
        protectedSettings: {
            script: encoded_script
        }
    }
}

output admin_username string = admin_username
output vm_name string = vm_name
output hostname string = public_IP.properties.dnsSettings.fqdn
output ssh_command string = 'ssh ${admin_username}@${public_IP.properties.dnsSettings.fqdn}'
