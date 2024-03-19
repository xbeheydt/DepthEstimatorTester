"""
Utils modules
"""

import requests
from pathlib import Path
from zipfile import ZipFile


def download(url: str, file_path: Path) -> None:
    """
    Downloads a URL content into a file (with large file support by streaming).

    Args:
        url (str): URL to download.
        file_path (str): Local file name to contain the data downloaded.
    """

    file_path = Path(file_path)
    if file_path.exists():
        raise FileExistsError(f"File \"{file_path}\" is already exists")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

        return file_path


def extract_archive(zipfile: str, path: str | None = None) -> None:
    """
    Extract a zip archive.

    Args:
        zipfile (str): Path to the zip archive.
        path (str): Path where to extract the archive.
    """
    with ZipFile(zipfile, "r") as z:
        z.extractall(path)
