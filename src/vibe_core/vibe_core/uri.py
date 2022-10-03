import os
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def is_local(url: str):
    """
    Check if uri refers to a local file
    """
    url_parsed = urlparse(url)
    return url_parsed.scheme in ("file", "")  # Possibly a local file


def local_uri_to_path(uri: str) -> str:
    """
    Maps 'file://' urls to paths. If the input is already a path, leave it as is.
    """
    if not is_local(uri):
        raise ValueError(f"Cannot convert remote URI {uri} to path")
    parsed = urlparse(uri)
    if parsed.scheme == "":  # Assume it is a path
        return uri
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.normpath(os.path.join(host, url2pathname(unquote(parsed.path))))


def uri_to_filename(uri: str) -> str:
    """
    Parse filename from uri
    """
    parsed_source_url = urlparse(uri)
    source_path = unquote(parsed_source_url.path)
    return os.path.basename(source_path)
