import mimetypes
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

from smb.SMBConnection import SMBConnection

from vibe_core.data import AssetVibe, DataVibe, Raster, gen_guid, gen_hash_id


def download_all_files(
    server_name: str,
    server_ip: str,
    server_port: int,
    username: str,
    password: str,
    share_name: str,
    directory_path: str,
    output_dir: Path,
) -> List[AssetVibe]:
    """Download all files under directory_path on the SMB share and return a list of AssetVibes."""
    # Establish a connection with the server
    conn = SMBConnection(
        username,
        password,
        "FarmVibes_SMB_Downloader",
        server_name,
        use_ntlm_v2=True,
        is_direct_tcp=True,
    )
    conn.connect(server_ip, server_port)

    # Collect all files in the directory as assets
    asset_list = []
    attributes = conn.getAttributes(share_name, directory_path)

    # Convert path to unix style
    directory_path = directory_path.replace("\\", "/")
    path = Path(directory_path)
    if attributes.isDirectory:
        crawl_directory(conn, share_name, path, asset_list, output_dir)
    else:
        download_asset(conn, share_name, path, asset_list, output_dir)
    return asset_list


def download_asset(
    conn: SMBConnection,
    share_name: str,
    filepath: Path,
    asset_list: List[AssetVibe],
    output_dir: Path,
):
    # Compute the output path
    if filepath.is_absolute():
        filepath = filepath.relative_to("/")
    output_path = output_dir.joinpath(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create an Asset type from the file
    with open(output_path, "wb") as asset_file:
        conn.retrieveFile(share_name, str(filepath), asset_file)
        asset = AssetVibe(
            reference=asset_file.name,
            type=mimetypes.guess_type(asset_file.name)[0],
            id=gen_guid(),
        )
    asset_list.append(asset)


def crawl_directory(
    conn: SMBConnection,
    share_name: str,
    dir_path: Path,
    asset_list: List[AssetVibe],
    output_dir: Path,
):
    """Recursively search through the file system starting at directory
    and download all files."""
    files = conn.listPath(share_name, str(dir_path))
    for file in files:
        if file.filename not in [".", ".."]:
            filepath = dir_path.joinpath(file.filename)
            if file.isDirectory:
                # Open subfolder
                crawl_directory(conn, share_name, filepath, asset_list, output_dir)
            else:
                # Download the file if it is an image
                mimetype = mimetypes.guess_type(str(filepath))[0]
                if mimetype and mimetype.startswith("image"):
                    download_asset(conn, share_name, filepath, asset_list, output_dir)


class CallbackBuilder:
    def __init__(
        self,
        server_name: str,
        server_ip: str,
        server_port: int,
        username: str,
        password: str,
        share_name: str,
        directory_path: str,
        bands: List[str],
    ):
        self.server_name = server_name
        self.server_ip = server_ip
        self.server_port = server_port
        self.username = username
        self.password = password
        self.share_name = share_name
        self.directory_path = directory_path
        self.bands = bands
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download(user_input: DataVibe) -> Dict[str, List[Raster]]:
            raster_assets = download_all_files(
                self.server_name,
                self.server_ip,
                self.server_port,
                self.username,
                self.password,
                self.share_name,
                self.directory_path,
                Path(self.tmp_dir.name),
            )
            bands = {name: index for index, name in enumerate(self.bands)}
            return {
                "rasters": [
                    Raster.clone_from(
                        user_input,
                        id=gen_hash_id(asset.id, user_input.geometry, user_input.time_range),
                        assets=[asset],
                        bands=bands,
                    )
                    for asset in raster_assets
                ]
            }

        return download

    def __del__(self):
        self.tmp_dir.cleanup()
