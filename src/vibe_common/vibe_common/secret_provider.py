import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from azure.core.credentials import TokenCredential
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import KeyVaultSecret, SecretClient
from dapr.clients import DaprClient
from hydra_zen import builds

from vibe_common.dapr import dapr_ready

CONNECTION_REFUSED_SUBSTRING = "connect: connection refused"
DAPR_WAIT_TIME_S = 30


class SecretProvider(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.expression = re.compile(r"^@SECRET\(([^,]*?), ([^,]*?)\)")

    def is_secret(self, value: str) -> bool:
        return self.expression.match(value) is not None

    @abstractmethod
    def _resolve_impl(self, value: Any) -> str:
        raise NotImplementedError

    def resolve(self, value: Any) -> str:
        if not (isinstance(value, str) and self.is_secret(value)):
            return value

        return self._resolve_impl(value)


class DaprSecretProvider(SecretProvider):
    def _resolve_impl(self, value: Any) -> str:
        while True:
            _, secret_name = self.expression.findall(value)[0]
            try:
                # dapr´s local file and local env vars secret stores do not allow
                # live update, that is, any update to a secret would require the
                # worker to be redeployed. So, we are using kubernetes secret store.
                # Even though Kubernetes supports multiple keys in a secret, secrets
                # added to the Kubernetes secret store via FarmVibes have a single
                # key whose name is the same as the secret name.
                return retrieve_dapr_secret("kubernetes", secret_name, secret_name)
            except Exception as e:
                stre = str(e)
                if CONNECTION_REFUSED_SUBSTRING in stre:
                    self.logger.info(
                        "dapr sidecar temporarily unavailable, "
                        f"will retry to resolve secret {value}"
                    )
                    # No need for exponential backoffs here. This is the k8s
                    # cluster misbehaving and it will return (hopefully
                    # quickly)
                    time.sleep(DAPR_WAIT_TIME_S)
                    continue
                raise ValueError(
                    f"Could not retrive secret {secret_name} from Dapr.\n Error message {stre}"
                )


class AzureSecretProvider(SecretProvider):
    def __init__(self, credential: Optional[TokenCredential] = None):
        super().__init__()
        self.__credential = credential

    @property
    def credential(self):
        if self.__credential is None:
            self.__credential = DefaultAzureCredential()

        return self.__credential

    def retrieve_from_keyvault(self, keyvault_name: str, secret_name: str) -> KeyVaultSecret:
        try:
            secret_client = SecretClient(
                vault_url=f"https://{keyvault_name}.vault.azure.net/", credential=self.credential
            )
            secret = secret_client.get_secret(secret_name)
        except ResourceNotFoundError as e:
            raise ValueError(f"Could not retrieve secret {secret_name}.\n Error message {str(e)}")
        except ServiceRequestError as e:
            raise ValueError(f"Invalid keyvault {keyvault_name}.\n Error message {str(e)}")

        return secret

    def _resolve_impl(self, value: Any) -> str:
        keyvault_name, secret_name = self.expression.findall(value)[0]
        secret = self.retrieve_from_keyvault(keyvault_name, secret_name)

        assert secret.value is not None

        return secret.value


@dapr_ready(dapr_wait_time_s=DAPR_WAIT_TIME_S)
def retrieve_dapr_secret(
    store_name: str,
    secret_name: str,
    key_name: str,
) -> str:
    """
    Using Dapr, retrieve a secret from a given secret store.

    Args:
        store_name: The name of the secret store from which to fetch the secret
        secret_name: The name of the secret to fetch
        key_name: The name of the key in the secret to fetch (Note: For secret stores that have
            multiple key-value pairs in a secret this would be the key to fetch. If the secret store
            supports only one key-value pair, this argument is the same as the `secret_name`.)

    Returns:
        The secret value
    """
    logger = logging.getLogger(f"{__name__}.retrieve_dapr_secret")
    with DaprClient() as dapr_client:
        key = dapr_client.get_secret(store_name, secret_name).secret[key_name]
        logger.info(f"Retrieving secret {secret_name} from store {store_name}")
        return key


def retrieve_keyvault_secret(
    keyvault_name: str, secret_name: str, cred: Optional[TokenCredential] = None
):
    cred = cred or DefaultAzureCredential()
    kv = SecretClient(keyvault_name, credential=cred)
    key = kv.get_secret(secret_name).value
    if key is None:
        raise ValueError(
            f"Could not find cosmos key with name {secret_name} on vault {keyvault_name}"
        )
    return key


DaprSecretConfig = builds(
    retrieve_dapr_secret,
    populate_full_signature=True,
    zen_dataclass={
        "module": "vibe_common.secret_provider",
        "cls_name": "DaprSecretConfig",
    },
)

KeyVaultSecretConfig = builds(
    retrieve_keyvault_secret,
    populate_full_signature=True,
    zen_dataclass={
        "module": "vibe_common.secret_provider",
        "cls_name": "KeyVaultSecretConfig",
    },
)

SecretProviderConfig = builds(
    SecretProvider,
    populate_full_signature=True,
    zen_dataclass={
        "module": "vibe_common.secret_provider",
        "cls_name": "SecretProviderConfig",
    },
)

DaprSecretProviderConfig = builds(
    DaprSecretProvider,
    populate_full_signature=True,
    builds_bases=(SecretProviderConfig,),
    zen_dataclass={
        "module": "vibe_common.secret_provider",
        "cls_name": "DaprSecretProviderConfig",
    },
)

AzureSecretProviderConfig = builds(
    AzureSecretProvider,
    populate_full_signature=True,
    builds_bases=(SecretProviderConfig,),
    zen_dataclass={
        "module": "vibe_common.secret_provider",
        "cls_name": "AzureSecretProviderConfig",
    },
)
