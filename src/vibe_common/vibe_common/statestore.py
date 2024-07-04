#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, List, Optional, Protocol, TypedDict

from vibe_common.constants import STATE_URL_TEMPLATE
from vibe_common.vibe_dapr_client import VibeDaprClient

LOGGER = logging.getLogger(__name__)
STATE_STORE = "statestore"
METADATA = {"partitionKey": "eywa"}


class TransactionOperation(TypedDict):
    key: str
    operation: str
    value: Optional[Any]


class StateStoreProtocol(Protocol):
    async def retrieve(self, key: str, traceparent: Optional[str] = None) -> Any: ...

    async def retrieve_bulk(
        self, keys: List[str], parallelism: int = 2, traceparent: Optional[str] = None
    ) -> List[Any]: ...

    async def store(self, key: str, obj: Any, traceparent: Optional[str] = None) -> bool: ...

    async def transaction(
        self, operations: List[TransactionOperation], traceparent: Optional[str] = None
    ) -> bool: ...


class StateStore(StateStoreProtocol):
    def __init__(
        self,
        state_store: str = STATE_STORE,
        partition_key: str = METADATA["partitionKey"],
    ):
        self.vibe_dapr_client = VibeDaprClient()
        self.state_store: str = state_store
        self.partition_key: str = partition_key
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def retrieve(self, key: str, traceparent: Optional[str] = None) -> Any:
        try:
            response = await self.vibe_dapr_client.get(
                STATE_URL_TEMPLATE.format(self.state_store, key),
                traceparent=traceparent,
                params={"metadata.partitionKey": METADATA["partitionKey"]},
            )

            return await self.vibe_dapr_client.response_json(response)
        except KeyError as e:
            raise KeyError(f"Key {key} not found") from e

    async def retrieve_bulk(
        self, keys: List[str], parallelism: int = 8, traceparent: Optional[str] = None
    ) -> List[Any]:
        """Retrieves keys in bulk.

        This only exists because our UI needs to display details about all
        workflows, and retrieving in bulk saves on round trips to the state
        store.
        """

        response = await self.vibe_dapr_client.post(
            url=STATE_URL_TEMPLATE.format(self.state_store, "bulk"),
            data={
                "keys": keys,
                "parallelism": parallelism,
            },
            traceparent=traceparent,
            params={"metadata.partitionKey": METADATA["partitionKey"]},
        )

        states = await self.vibe_dapr_client.response_json(response)

        if len(states) != len(keys):
            keyset = set(keys)
            for state in states:
                keyset.remove(state[0])
            raise KeyError(f"Failed to retrieve keys {keyset} from state store.")
        return [state["data"] for state in states]

    async def store(self, key: str, obj: Any, traceparent: Optional[str] = None) -> None:
        response = await self.vibe_dapr_client.post(
            STATE_URL_TEMPLATE.format(self.state_store, ""),
            data=[
                {
                    "key": key,
                    "value": self.vibe_dapr_client.obj_json(obj),
                    "metadata": {"partitionKey": self.partition_key},
                }
            ],
            traceparent=traceparent,
        )
        assert response.ok, "Failed to store state, but underlying method didn't capture it"

    async def transaction(
        self, operations: List[TransactionOperation], traceparent: Optional[str] = None
    ) -> None:
        queries = [
            {
                "operation": o["operation"],
                "request": {
                    "key": o["key"],
                    "value": self.vibe_dapr_client.obj_json(o["value"]),
                },
            }
            for o in operations
        ]
        await self.vibe_dapr_client.post(
            url=STATE_URL_TEMPLATE.format(self.state_store, "transaction"),
            data={
                "operations": queries,
                "metadata": {"partitionKey": self.partition_key},
            },
            traceparent=traceparent,
        )
