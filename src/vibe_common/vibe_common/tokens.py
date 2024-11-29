# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, cast
from urllib.parse import urljoin, urlparse

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobClient,
    BlobSasPermissions,
    BlobServiceClient,
    UserDelegationKey,
    generate_blob_sas,
)


class StorageUserKey(ABC):
    @abstractmethod
    def is_valid(self) -> bool:
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def get_access_key(self) -> Union[UserDelegationKey, str]:
        raise NotImplementedError("Subclass needs to implement this")


class StorageUserKeyCredentialed(StorageUserKey):
    delegation_key: UserDelegationKey
    key_expiration: Optional[datetime]
    sas_expiration: timedelta

    def __init__(
        self,
        url: str,
        sas_expiration: timedelta,
        key_lease_time: timedelta,
        credential: Optional[TokenCredential] = None,
    ):
        self.sas_expiration = sas_expiration
        self.key_lease_time = key_lease_time
        self.credential = DefaultAzureCredential() if credential is None else credential
        self.storage_url = self._get_storage_url(url)
        self.client = None

        # Update expiration and delegation keys
        self._generate()

    def _get_storage_url(self, url: str) -> str:
        return urlparse(url.rstrip("/")).netloc

    def _get_client(self):
        if not self.client:
            self.client = BlobServiceClient(self.storage_url, self.credential)

        return self.client

    def is_valid(self) -> bool:
        if not self.key_expiration:
            return False
        return datetime.utcnow() + self.sas_expiration < self.key_expiration

    def _generate(self):
        self.key_expiration = datetime.utcnow() + self.key_lease_time
        client = self._get_client()
        self.delegation_key = client.get_user_delegation_key(datetime.utcnow(), self.key_expiration)

    def get_access_key(self) -> Union[UserDelegationKey, str]:
        if not self.is_valid():
            self._generate()
        return self.delegation_key


class StorageUserKeyConnectionString(StorageUserKey):
    def __init__(
        self,
        sas_expiration: timedelta,
        key_lease_time: timedelta,
        connection_string: str,
    ):
        self.connection_string = connection_string
        self.client = None

    def _get_client(self):
        if not self.client:
            self.client = BlobServiceClient.from_connection_string(self.connection_string)

        return self.client

    def is_valid(self) -> bool:
        return True

    def get_access_key(self) -> Union[UserDelegationKey, str]:
        client = self._get_client()
        return client.credential.account_key


class BlobTokenManager(ABC):
    sas_expiration_days: int
    lease_time_multiplier: int
    user_key_cache: Dict[str, StorageUserKey] = {}

    def __init__(
        self,
        sas_expiration_days: int = 1,
        lease_time_ratio: int = 2,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sas_expiration = timedelta(days=sas_expiration_days)
        self.lease_time_ratio = lease_time_ratio
        self.key_lease_time = self.lease_time_ratio * self.sas_expiration

    @abstractmethod
    def _get_storage_user_key(
        self,
        url: str,
        sas_expiration: timedelta,
        key_lease_time: timedelta,
    ) -> StorageUserKey:
        raise NotImplementedError("Subclass needs to implement this")

    def _get_user_key(self, url: str, account_name: str) -> StorageUserKey:
        if account_name not in self.user_key_cache:
            self.logger.debug(f"Creating a new user key for account {account_name}")
            storage_user_key = self._get_storage_user_key(
                url, self.sas_expiration, self.key_lease_time
            )

            self.user_key_cache[account_name] = storage_user_key

        return self.user_key_cache[account_name]

    @abstractmethod
    def _get_token(self, blob_client: BlobClient):
        raise NotImplementedError("Subclass needs to implement this")

    def sign_url(self, url: str) -> str:
        blob_client = BlobClient.from_blob_url(blob_url=url)
        sas_token = self._get_token(blob_client)
        return f"{urljoin(url, urlparse(url).path)}?{sas_token}"


class BlobTokenManagerCredentialed(BlobTokenManager):
    def __init__(
        self,
        sas_expiration_days: int = 1,
        lease_time_ratio: int = 2,
        credential: Optional[TokenCredential] = None,
    ):
        super().__init__(sas_expiration_days, lease_time_ratio)
        self.credential = DefaultAzureCredential() if credential is None else credential

    def _get_storage_user_key(
        self,
        url: str,
        sas_expiration: timedelta,
        key_lease_time: timedelta,
    ) -> StorageUserKey:
        return StorageUserKeyCredentialed(
            url,
            sas_expiration,
            key_lease_time,
            credential=self.credential,
        )

    def _get_token(
        self,
        blob_client: BlobClient,
    ):
        account_name: str = cast(str, blob_client.account_name)
        container_name: str = blob_client.container_name
        blob_name: str = blob_client.blob_name

        start = datetime.utcnow()
        end = start + self.sas_expiration
        user_delegation_key = cast(
            UserDelegationKey, self._get_user_key(blob_client.url, account_name).get_access_key()
        )

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            user_delegation_key=user_delegation_key,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            start=start,
            expiry=end,
        )
        return sas_token


class BlobTokenManagerConnectionString(BlobTokenManager):
    def __init__(
        self,
        connection_string: str,
        sas_expiration_days: int = 1,
        lease_time_ratio: int = 2,
    ):
        super().__init__(sas_expiration_days, lease_time_ratio)
        self.connection_string = connection_string

    def _get_storage_user_key(
        self,
        url: str,
        sas_expiration: timedelta,
        key_lease_time: timedelta,
    ) -> StorageUserKey:
        return StorageUserKeyConnectionString(
            sas_expiration,
            key_lease_time,
            self.connection_string,
        )

    def _get_token(
        self,
        blob_client: BlobClient,
    ):
        account_name: str = cast(str, blob_client.account_name)
        container_name: str = blob_client.container_name
        blob_name: str = blob_client.blob_name

        start = datetime.utcnow()
        end = start + self.sas_expiration
        account_key = cast(str, self._get_user_key(blob_client.url, account_name).get_access_key())
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            account_key=account_key,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            start=start,
            expiry=end,
        )
        return sas_token
