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
    description='Python package for Granger causality test with nonlinear (neural networks) forecasting methods.',
    version='v1.1.1',
    long_description=README,
    url='https://github.com/mrosol/Nonlincausality',
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=['numpy', 'pandas','scipy', 'keras', 'statsmodels', 'tensorflow', 'matplotlib'],
    keywords='Granger causality neural networks nonlinear forecasting signals',
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