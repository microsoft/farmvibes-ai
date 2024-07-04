import os
import shutil


def create_flat_archive(directory_path: str, archive_name: str) -> str:
    """Create a flat file directory zip archive containing all files under the given directory.
    Traverses subdirectories to find all files.

    Args:
        directory_path: directory to archive
        archive_name: name to give the archive (without .zip extension)

    Returns:
        Path to zipped archive containing all files at the root level
    """
    files_to_move = []
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            files_to_move.append(filepath)

    archive_dir = os.path.join(directory_path, archive_name)
    os.mkdir(archive_dir)
    for file in files_to_move:
        shutil.move(file, archive_dir)

    archive_path = os.path.join(directory_path, archive_name)
    return shutil.make_archive(archive_path, "zip", archive_dir)
