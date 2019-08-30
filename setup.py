#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import os

packages = find_packages()
setup(
    name = "CosmoloPy",
    version = "0.1.105",
    packages = packages,
    install_requires = ['numpy', 'scipy',],
    tests_require = ['matplotlib'],

    # metadata for upload to PyPI
    author = "Roban Hultman Kramer",
    author_email = "robanhk@gmail.com",
    description = "a cosmology package for Python.",
    url = "http://roban.github.com/CosmoloPy/",   # project home page
    keywords = ("astronomy cosmology cosmological distance density galaxy" +
                "luminosity magnitude reionization Press-Schechter Schecter"),
    license = "MIT",
    long_description = \
      
"""CosmoloPy is a package of cosmology routines built on NumPy/SciPy.

Capabilities include
--------------------

`cosmolopy.density`
  Various cosmological densities.

`cosmolopy.distance`
  Various cosmological distance measures.

`cosmolopy.luminosityfunction`
  Galaxy luminosity functions (Schecter functions).

`cosmolopy.magnitudes`
  Conversion in and out of the AB magnitude system.

`cosmolopy.parameters`
  Pre-defined sets of cosmological parameters (e.g. from WMAP).

`cosmolopy.perturbation`
  Perturbation theory and the power spectrum.

`cosmolopy.reionization`
  The reionization of the IGM.
  
""",
    classifiers = ['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.7',
                   'Topic :: Astronomy']
)
