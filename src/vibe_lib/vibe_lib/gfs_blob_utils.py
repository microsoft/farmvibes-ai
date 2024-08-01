# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime

"""
Utilities for dealing with NOAA GFS data in Azure Blob Store
"""

# Blob container URI for GFS data
NOAA_BLOB_URI = "https://noaagfs.blob.core.windows.net/gfs"


def get_sas_uri(sas_token: str) -> str:
    return "{uri}?{sas}".format(uri=NOAA_BLOB_URI, sas=sas_token)


def blob_url_from_offset(publish_date: datetime, offset: int) -> str:
    date_str = publish_date.date().isoformat().replace("-", "")
    hour_str = str(publish_date.hour).rjust(2, "0")
    offset_str = str(offset).rjust(3, "0")
    return "gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f{offset}".format(
        date=date_str, hour=hour_str, offset=offset_str
    )
