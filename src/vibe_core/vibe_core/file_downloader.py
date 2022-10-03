import logging
import mimetypes
import os
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
REQUEST_RETRIES = 5
REQUEST_BACKOFF = 0.3
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 30

LOGGER = logging.getLogger(__name__)


def retry_session() -> requests.Session:
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
    """Method to verify is a URL does not raise HTTP or Connection errors"""
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
