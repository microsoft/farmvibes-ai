import os
from datetime import datetime
from unittest.mock import Mock, patch
from zipfile import ZipFile

import pytest
from shapely import geometry as shpg

from vibe_core.data import DownloadedSentinel1Product, Sentinel1Product
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import generate_sentinel1_blob_path

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "download_sentinel1_grd.yaml")
FULL_PRODUCT_NAME = "S1B_IW_GRDH_1SDV_20200508T141252_20200508T141322_021491_028CDD_C1D0"


class MockBlob:
    def __init__(self, name: str):
        self.name = name

    def __getitem__(self, key: str):
        return getattr(self, key)


def fake_download(_, file_path: str):
    with open(os.path.join(file_path), "w") as f:
        f.write("🌎")


@pytest.mark.parametrize("product_name", ("complete", "incomplete"))
@patch("vibe_core.file_downloader.download_file")
@patch("vibe_lib.planetary_computer.get_sentinel1_scene_name")
@patch("vibe_lib.planetary_computer.get_sentinel1_scene_files")
@patch("vibe_lib.planetary_computer.get_sentinel1_container_client")
def test_op(
    get_s1_client: Mock,
    s1_scene_files: Mock,
    s1_scene_name: Mock,
    download_file: Mock,
    product_name: str,
):
    s1_scene_name.return_value = FULL_PRODUCT_NAME
    download_file.side_effect = fake_download
    geom = shpg.box(0, 0, 1, 1)
    fake_input = Sentinel1Product(
        id="1",
        time_range=(datetime.now(), datetime.now()),
        geometry=shpg.mapping(geom),
        assets=[],
        product_name=FULL_PRODUCT_NAME,
        orbit_number=0,
        relative_orbit_number=0,
        orbit_direction="",
        platform="",
        extra_info={},
        sensor_mode="",
        polarisation_mode="",
    )
    blob_path = generate_sentinel1_blob_path(fake_input)
    s1_scene_files.return_value = [
        MockBlob(f"{blob_path}/fake.txt"),
        MockBlob(f"{blob_path}/fake_dir/fake2.txt"),
    ]
    op_tester = OpTester(CONFIG_PATH)
    if product_name == "incomplete":
        fake_input.product_name = FULL_PRODUCT_NAME[:-4]
    out = op_tester.run(sentinel_product=fake_input)
    key = "downloaded_product"
    assert key in out
    product = out[key]
    assert isinstance(product, DownloadedSentinel1Product)
    zip_path = product.get_zip_asset().local_path
    assert os.path.basename(zip_path) == f"{FULL_PRODUCT_NAME}.zip"
    base_dir = f"{FULL_PRODUCT_NAME}.SAFE"
    with ZipFile(zip_path) as zf:
        il = zf.infolist()
        assert len(il) == 4
        assert f"{base_dir}/" == il[0].filename
        assert f"{base_dir}/fake_dir/" == il[1].filename
        assert f"{base_dir}/fake.txt" == il[2].filename
        assert f"{base_dir}/fake_dir/fake2.txt" == il[3].filename
        with zf.open(il[2]) as f:
            content = f.read()
        with zf.open(il[3]) as f:
            content2 = f.read()
    assert content.decode("utf-8") == content2.decode("utf-8") == "🌎"
