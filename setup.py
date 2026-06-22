"""Packaging shim.

The compiled STRUMPACK extension (`_strumpack_ext*.so`) is produced out-of-band
by CMake (per platform: cpu / rocm / cuda) and dropped into the package dir
before building the wheel. Declaring the distribution as binary forces a
platform-tagged wheel (not a pure-python one) and bundles the .so.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # force a platform wheel, include the .so
        return True


setup(distclass=BinaryDistribution)
