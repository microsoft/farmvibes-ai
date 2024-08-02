# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from datetime import datetime, timezone

from shapely.geometry import Polygon

from vibe_core.client import FarmvibesAiClient, get_default_vibe_client


def main():
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)

    polygon_coords = [
        (-88.062073563448919, 37.081397673802059),
        (-88.026349330507315, 37.085463858128762),
        (-88.026349330507315, 37.085463858128762),
        (-88.012445388773259, 37.069230099135126),
        (-88.035931592028305, 37.048441375086092),
        (-88.068120429075847, 37.058833638440767),
        (-88.062073563448919, 37.081397673802059),
    ]

    polygon = Polygon(polygon_coords)
    start_date = datetime(year=2021, month=2, day=1, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=2, day=11, tzinfo=timezone.utc)
    client: FarmvibesAiClient = get_default_vibe_client()

    LOGGER.info(f"Successfully obtained a FarmVibes.AI client (addr={client.baseurl})")
    LOGGER.info(f"available workflows: {client.list_workflows()}")

    LOGGER.info("Running helloworld workflow...")
    run = client.run(
        "helloworld", "test_hello", geometry=polygon, time_range=(start_date, end_date)
    )

    try:
        run.block_until_complete(30)
        LOGGER.info(f"Successfully executed helloworld workflow. Result {run}")
    except RuntimeError as e:
        LOGGER.error(f"Failed to execute workflow. Reason: {e}")
        raise


if __name__ == "__main__":
    main()
