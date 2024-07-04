import json
import logging
from functools import partial
from typing import Any, Mapping, Optional

from aiohttp import ClientResponse, ClientSession
from aiohttp_retry import ExponentialRetry, RetryClient

from vibe_common.constants import TRACEPARENT_HEADER_KEY
from vibe_common.dapr import handle_aiohttp_timeout, process_dapr_response
from vibe_core.data.json_converter import dump_to_json

MAX_SESSION_ATTEMPTS = 10
MAX_TIMEOUT_S = 30
MAX_DIRECT_INVOKE_TRIES = 3

METADATA = {"partitionKey": "eywa"}

"""
This is an implementation of a Dapr HTTP client that currently support Dapr service invocation
and state management through HTTP.
"""


class VibeDaprClient:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _build_client(self) -> RetryClient:
        session = ClientSession()
        retry_options = ExponentialRetry(
            attempts=MAX_SESSION_ATTEMPTS,
            max_timeout=MAX_TIMEOUT_S,
            statuses={400, 500, 502, 503, 504},
        )
        retry_client = RetryClient(client_session=session, retry_options=retry_options)
        return retry_client

    async def get(
        self,
        url: str,
        traceparent: Optional[str],
        params: Optional[Mapping[str, str]] = None,
    ) -> ClientResponse:
        async with self._build_client() as session:
            try:
                response = await session.get(
                    url, headers={"traceparent": traceparent} if traceparent else {}, params=params
                )
                await handle_aiohttp_timeout(response)
                return await process_dapr_response(response)
            except KeyError:
                raise
            except Exception:
                self.logger.exception(f"Failed to process request for {url}")
                raise RuntimeError(f"dapr failed to process request for {url}")

    async def post(
        self,
        url: str,
        data: Any,
        traceparent: Optional[str],
        params: Optional[Mapping[str, str]] = None,
    ) -> ClientResponse:
        if url.endswith("/"):
            url = url[:-1]

        tries: int = 0

        while True:
            async with self._build_client() as session:
                try:
                    headers = {"Content-Type": "application/json"}
                    if traceparent:
                        headers[TRACEPARENT_HEADER_KEY] = traceparent
                    response = await session.post(
                        url,
                        data=self._dumps(data),
                        headers=headers,
                        params=params,
                    )
                    await handle_aiohttp_timeout(response)
                    return await process_dapr_response(response)
                except RuntimeError as e:
                    if "ERR_DIRECT_INVOKE" in str(e):
                        tries += 1
                        self.logger.warning(
                            f"ERR_DIRECT_INVOKE raised by Dapr, "
                            f"retrying ({tries}/{MAX_DIRECT_INVOKE_TRIES})"
                        )
                        if tries >= MAX_DIRECT_INVOKE_TRIES:
                            self.logger.exception(f"Failed to process request for {url}")
                            raise
                except Exception:
                    self.logger.exception(f"Failed to process request for {url}")
                    raise RuntimeError(f"dapr failed to process request for {url}")

    def obj_json(self, obj: Any, **kwargs: Any) -> Any:
        """JSON representation of object `obj` encoding floats as strings.

        Unfortunately Dapr's JSON deserializer clips floating point precision
        so floats are encoded as strings

        Args:
            obj: the object to be converted
            kwargs: optional keyword arguments passed to `_dumps`

        Returns:
            Object `obj` represented as JSON
        """
        return json.loads(self._dumps(obj, **kwargs), parse_float=lambda f_as_s: f_as_s)

    async def response_json(self, response: ClientResponse) -> Any:
        """Loads a JSON from a `ClientResponse`.

        Because floats are encoded as strings before being sent to Dapr due to the truncation that
        occurs in the Dapr sidecar when using its HTTP API, this method decodes any string that
        can be parsed as a float into a Python float.

        Args:
            response: The `ClientResponse` object with our data

        Returns:
            The JSON of our response, with floats correctly decoded as floats
        """
        return await response.json(loads=partial(json.loads, object_hook=_decode))

    def _dumps(self, obj: Any, **kwargs: Any) -> str:
        return dump_to_json(obj, **kwargs)


def _decode(obj: Any) -> Any:
    """Returns the given decoded JSON object with all string values that can be parsed as floats as
    Python floats.

    This function covers all possible valid JSON objects as valid JSON values are strings, objects
    (Python dict), arrays (Python list), numbers (Python int/float), or the literals true (Python
    True), false (Python False), or null (Python None)):
    https://www.rfc-editor.org/rfc/rfc8259#section-3

    Args:
        obj: A decoded JSON object

    Returns:
        The same decoded JSON object with all string values that can be parsed as floats as floats
    """
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    elif isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode(v) for v in obj]
    else:
        return obj
