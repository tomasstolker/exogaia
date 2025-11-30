"""
Initialization of the ``exogaia`` package.
"""

import json
import os
import socket
import urllib.request

from exogaia.data import EpochAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.model import BinaryModel
from exogaia.priors import LogUniformPrior, NormalPrior, SinPrior, UniformPrior
from exogaia.results import FitResults
from exogaia.sampler import NestedSampler

from ._version import __version__, __version_tuple__

__author__ = "Tomas Stolker"
__license__ = "MIT"
__maintainer__ = "Tomas Stolker"
__email__ = "stolker@strw.leidenuniv.nl"

print("========\nexogaia\n========")

# Check if there is a new version available

EXOGAIA_VERSION = (
    f"{__version_tuple__[0]}." f"{__version_tuple__[1]}." f"{__version_tuple__[2]}"
)

PYPI_URL = "https://pypi.org/pypi/exogaia/json"

try:
    with urllib.request.urlopen(PYPI_URL, timeout=1.0) as open_url:
        url_content = open_url.read()
        url_data = json.loads(url_content)
        pypi_version = url_data["info"]["version"]

except (urllib.error.URLError, socket.timeout):
    pypi_version = None

print(f"\nVersion: {__version__}")

if pypi_version is not None:
    pypi_split = pypi_version.split(".")
    current_split = EXOGAIA_VERSION.split(".")

    new_major = (pypi_split[0] == current_split[0]) & (pypi_split[1] > current_split[1])

    new_minor = (
        (pypi_split[0] == current_split[0])
        & (pypi_split[1] == current_split[1])
        & (pypi_split[2] > current_split[2])
    )

    if new_major | new_minor:
        print(f"\nA new version ({pypi_version}) is available!")
        print("Update exogaia by running:")
        print("pip install --upgrade exogaia")
