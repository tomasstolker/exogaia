"""
Module with the ``exogaia`` tool.
"""

import json
import socket
import urllib.request

from astroquery.gaia import Gaia
from typeguard import typechecked

from ._version import __version__, __version_tuple__


# No limit on the number of rows with a Gaia query
Gaia.ROW_LIMIT = -1


class ExoGaia:
    """
    Main class.
    """

    @typechecked
    def __init__(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        print("========\nexogaia\n========")

        # Check if there is a new version available

        exogaia_version = (
            f"{__version_tuple__[0]}."
            f"{__version_tuple__[1]}."
            f"{__version_tuple__[2]}"
        )

        try:
            pypi_url = "https://pypi.org/pypi/exogaia/json"

            with urllib.request.urlopen(pypi_url, timeout=1.0) as open_url:
                url_content = open_url.read()
                url_data = json.loads(url_content)
                pypi_version = url_data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            pypi_version = None

        print(f"\nVersion: {__version__}")

        if pypi_version is not None:
            pypi_split = pypi_version.split(".")
            current_split = exogaia_version.split(".")

            new_major = (pypi_split[0] == current_split[0]) & (
                pypi_split[1] > current_split[1]
            )

            new_minor = (
                (pypi_split[0] == current_split[0])
                & (pypi_split[1] == current_split[1])
                & (pypi_split[2] > current_split[2])
            )

            if new_major | new_minor:
                print(f"\nA new version ({pypi_version}) is available!")
                print("Want to stay informed about updates?")
                print("Please have a look at the Github page:")
                print("https://github.com/tomasstolker/exogaia")
