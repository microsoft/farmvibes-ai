# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import logging
from functools import partial, wraps
from typing import Any, Callable, overload

from aiohttp import ClientResponse
from dapr.clients import DaprClient
from dapr.conf import settings

from vibe_common.constants import SERVICE_INVOCACATION_URL_PATH, STATE_URL_PATH

MAX_TIMEOUT_TRIES = 3
DAPR_WAIT_TIME_S = 90


def dapr_ready_decorator(
    func: Callable[..., Any], dapr_wait_time_s: int = DAPR_WAIT_TIME_S
) -> Callable[..., Any]:
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):  # type: ignore
            with DaprClient() as dapr_client:
                logger = logging.getLogger(f"{__name__}.wait_dapr")
                logger.info(f"Waiting {dapr_wait_time_s} seconds for dapr to be ready")
                try:
                    dapr_client.wait(dapr_wait_time_s)
                except Exception:
                    logger.exception("dapr is not ready")
                    raise
                logger.info("dapr is ready.")
            return await func(*args, **kwargs)
    else:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with DaprClient() as dapr_client:
                logger = logging.getLogger(f"{__name__}.wait_dapr")
                logger.info(f"Waiting {dapr_wait_time_s} seconds for dapr to be ready")
                try:
                    dapr_client.wait(dapr_wait_time_s)
                except Exception:
                    logger.exception("dapr is not ready")
                    raise
                logger.info("dapr is ready.")
            return func(*args, **kwargs)

    return wrapper


@overload
def dapr_ready(func: None = None, *, dapr_wait_time_s: int = DAPR_WAIT_TIME_S) -> Any: ...


@overload
def dapr_ready(func: Callable[..., Any]) -> Callable[..., Any]: ...


def dapr_ready(func: Any = None, *, dapr_wait_time_s: int = DAPR_WAIT_TIME_S) -> Any:
    if func is None:
        return partial(dapr_ready_decorator, dapr_wait_time_s=dapr_wait_time_s)
    else:
        return dapr_ready_decorator(func, dapr_wait_time_s=dapr_wait_time_s)


def process_dapr_state_response(response: ClientResponse) -> ClientResponse:
    if not response.ok:
        if response.status == 400:
            raise RuntimeError("State store is not configured")
        elif response.status == 404:
            raise KeyError(f"Key specified in {response.url} not found")
    if response.request_info.method == "GET" and response.status == 204:
        # https://docs.dapr.io/reference/api/state_api/#http-response-1
        raise KeyError(f"Key specified in {response.url} not found")
    return response


async def process_dapr_service_invocation_response(
    response: ClientResponse,
) -> ClientResponse:
    if not response.ok:
        if response.status == 400:
            raise RuntimeError("Method name not given for service invocation.")
        elif response.status == 403:
            raise RuntimeError(f"Invocation forbidden by access control for {response.url}")
        elif response.status == 500:
            content = await response.read()
            raise RuntimeError(f"Response 500 for {response.url} -- response body: {content}")
    return response


async def process_dapr_response(response: ClientResponse) -> ClientResponse:
    if response.url.host != settings.DAPR_RUNTIME_HOST:
        logging.warning("This url is not a response from Dapr: {response.url.host}")
        return response

    if response.url.path.startswith(STATE_URL_PATH):
        return process_dapr_state_response(response)
    elif response.url.path.startswith(SERVICE_INVOCACATION_URL_PATH):
        return await process_dapr_service_invocation_response(response)
    else:
        logging.warning(
            "We only handle Dapr responses for state management and service invocation. "
            "Response URL = {response.url}"
        )
        return response


async def handle_aiohttp_timeout(response: ClientResponse) -> ClientResponse:
    logger = logging.getLogger(f"{__name__}.handle_aiohttp_timeout")
    tries: int = 0
    while True:
        try:
            await response.read()
            return await process_dapr_response(response)
        except asyncio.TimeoutError:
            tries += 1
            logger.warning(
                f"Timeout interacting with Dapr via HTTP, "
                f"retrying ({tries}/{MAX_TIMEOUT_TRIES})"
            )
            if tries >= MAX_TIMEOUT_TRIES:
                raise
