from contextlib import suppress

import matplotlib as mpl

with suppress(ModuleNotFoundError, ImportError):
    from downscaler._version import __version__

from downscaler.constants import constants
from downscaler.constants import IFT, RFT

# this is a workaround since the backend on a Mac machine needs to be "TKAgg"
# but this in turn causes problems on a linux system so there we just use the
# default backend
try:
    mpl.use("TkAgg")
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):
    mpl.use("agg")
    import matplotlib.pyplot as plt

CONSTANTS = constants()

# caching enabled per default
USE_CACHING = True
