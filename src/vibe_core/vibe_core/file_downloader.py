import logging
import mimetypes
import os
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
"""Size of the chunks to read from the server per request."""

REQUEST_RETRIES = 5
"""Number of retries to perform when a request fails."""

REQUEST_BACKOFF = 0.3
"""Back-off factor to apply between retries."""

CONNECT_TIMEOUT_S = 30
"""Time in seconds to wait for connection to the server before aborting."""

READ_TIMEOUT_S = 30
"""Time in seconds for each chunk read from the server."""

LOGGER = logging.getLogger(__name__)


def retry_session() -> requests.Session:
    """Creates a session with retry support.

    This method creates a requests.Session object with retry support
    configured to retry failed requests up to :const:`REQUEST_RETRIES` times
    with a :const:`REQUEST_BACKOFF` time back-off factor.

    :return: A configured requests.Session object
    """
    session = requests.Session()

    retry = Retry(
        total=REQUEST_RETRIES,
        read=REQUEST_RETRIES,
        connect=REQUEST_RETRIES,
        backoff_factor=REQUEST_BACKOFF,
    )

    # Had to ignore the type as urlib is loaded dinamically
    # details here (https://github.com/microsoft/pylance-release/issues/597)
    adapter = HTTPAdapter(max_retries=retry)  # type: ignore
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def build_file_path(dir_name: str, file_name: str, type: str = "") -> str:
    """
    Builds the full file path by combining the directory name, file name and
    optional type to infer the file extension.

    :param dir_name: Name of the directory.

    :param file_name: Name of the file.

    :param type: Type of the file (default is empty).

    :return: The full file path.
    """
    extension = mimetypes.guess_extension(type)
    if not extension:
        LOGGER.info(f"File extension could no be inferred with type {type}. Using no extension.")
        extension = ""

    file_path = os.path.join(dir_name, f"{file_name}{extension}")
    return file_path


def download_file(
    url: str,
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    connect_timeout: float = CONNECT_TIMEOUT_S,
    read_timeout: float = READ_TIMEOUT_S,  # applies per chunk
    **kwargs: Any,
) -> str:
    """Downloads a file from a given URL to the given file path.

    The download is done using a retry session, to handle connection errors.

    :param url: URL of the file to download.

    :param file_path: Path where the file will be saved.

    :param chunk_size: Amount of data to read from the server per request
        (defaults to :const:`CHUNK_SIZE`).

    :param connect_timeout: Time in seconds to wait for connection to the server before aborting
        (defaults to :const:`CONNECT_TIMEOUT_S`).

    :param read_timeout: Time in seconds for each chunk read from the server
        (defaults to :const:`READ_TIMEOUT_S`).

    :param kwargs: Additional keyword arguments to be passed to the request library call.

    :return: Path of the saved file.
    """

    session = retry_session()

    try:
        with session.get(url, stream=True, timeout=(connect_timeout, read_timeout), **kwargs) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                return file_path
    except requests.ConnectionError:
        LOGGER.exception(f"Connection error when downloading remote asset {url}")
        raise
    except requests.HTTPError as err:
        LOGGER.exception(
            f"HTTP error (code {err.response.status_code}) when downloading remote asset {url}"
        )
        raise


def verify_url(
    url: str,
    connect_timeout: float = CONNECT_TIMEOUT_S,
    **kwargs: Any,
) -> bool:
    """Verifies the validity of a given URL.

    This method attempts to connect to the specified url and verifies
    that it does not raise any HTTP or Connection errors.

    :param url: The URL to check.

    :param connect_timeout: Timeout when attempting to connect to the specified url.
        Defaults to the value of :const:`CONNECT_TIMEOUT_S`.

    :param kwargs: Additional keyword arguments to pass to the requests.get call.

    :return: True if the URL is valid, False otherwise.
    """

    status = True
    session = retry_session()
    try:
        with session.get(url, stream=True, timeout=connect_timeout, **kwargs) as r:
            r.raise_for_status()
    except requests.ConnectionError:
        LOGGER.warning(f"Connection error when verifying remote asset {url}")
        status = False
    except requests.HTTPError as err:
        LOGGER.warning(
            f"HTTP error (code {err.response.status_code}) when verifying remote asset {url}"
        )
        status = False
    return status
