# -*- coding: utf-8 -*-

from .I2MC import I2MC, get_fixations
from .version import __version__, __url__, __author__, __email__, __description__
from . import plot


__all__ = [
    "I2MC",
    "get_fixations",
    "plot",
]
