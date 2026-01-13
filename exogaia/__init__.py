"""
Initialization of the ``exogaia`` package.
"""

import json
import os
import socket
import urllib.request

import matplotlib as mpl

from exogaia.data import EpochAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.limits import CompletenessMap
from exogaia.models import BinaryModel
from exogaia.priors import (
    FixedPrior,
    LogUniformPrior,
    NormalPrior,
    SinPrior,
    UniformPrior,
)
from exogaia.results import FitResults
from exogaia.sampler import MCMCSampler, NestedSampler

from ._version import __version__, __version_tuple__

__author__ = "Tomas Stolker"
__license__ = "MIT"
__maintainer__ = "Tomas Stolker"
__email__ = "stolker@strw.leidenuniv.nl"

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
        print(f"\nexogaia v{pypi_version} is available!")

# Set Matplotlib style

# Font settings
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"

# Tick visibility
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.bottom"] = True
mpl.rcParams["ytick.left"] = True
mpl.rcParams["ytick.right"] = True

mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["xtick.minor.bottom"] = True
mpl.rcParams["ytick.minor.left"] = True
mpl.rcParams["ytick.minor.right"] = True

# Tick sizes
mpl.rcParams["xtick.major.size"] = 5
mpl.rcParams["ytick.major.size"] = 5
mpl.rcParams["xtick.minor.size"] = 3
mpl.rcParams["ytick.minor.size"] = 3

# Tick widths
mpl.rcParams["xtick.major.width"] = 1
mpl.rcParams["ytick.major.width"] = 1
mpl.rcParams["xtick.minor.width"] = 1
mpl.rcParams["ytick.minor.width"] = 1

# Tick directions
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

# Minor tick visibility
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True

# Figure size (in inches)
mpl.rcParams["figure.figsize"] = (5, 3)

# Axes settings
mpl.rcParams["axes.axisbelow"] = False
mpl.rcParams["axes.edgecolor"] = "black"
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["axes.titlesize"] = "large"
mpl.rcParams["axes.labelsize"] = "large"

# Savefig options
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
