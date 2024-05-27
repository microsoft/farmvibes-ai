"""Utility functions for working with URIs."""

import os
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def is_local(url: str):
    """Check if uri refers to a local file.

    Args:
        url: The URL to check.

    Returns:
        True if the URL refers to a local file, False otherwise.
    """
    url_parsed = urlparse(url)
    return url_parsed.scheme in ("file", "")  # Possibly a local file


def local_uri_to_path(uri: str) -> str:
    """Map 'file://' urls to paths. If the input is already a path, leave it as is.

    Args:
        uri: The URI to convert.

    Returns:
        The path corresponding to the URI.

    Raises:
        ValueError: If the URI is not local.
    """
    if not is_local(uri):
        raise ValueError(f"Cannot convert remote URI {uri} to path")
    parsed = urlparse(uri)
    if parsed.scheme == "":  # Assume it is a path
        return uri
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.normpath(os.path.join(host, url2pathname(unquote(parsed.path))))


def uri_to_filename(uri: str) -> str:
    """Parse the filename from an URI.

    Args:
        uri: The URI to convert.

    Returns:
        The filename associated with the URI.
    """
    parsed_source_url = urlparse(uri)
    source_path = unquote(parsed_source_url.path)
    return os.path.basename(source_path)
