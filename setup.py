# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 22:52:23 2020

@author: MSc. Maciej Rosoł
contact: mrosol5@gmail.com
"""


import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Maciej Rosoł",
    author_email="mrosol5@gmail.com",
    name='nonlincausality',
    license="MIT",
    description='Python package for Granger causality test with nonlinear forecasting methods.',
    version='v0.0.1',
    long_description=README,
    url='https://github.com/mrosol/nonlincausality',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['scipy', 'keras', 'statsmodels', 'tensorflow', 'matplotlib'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)