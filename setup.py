# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 22:52:23 2020

@author: MSc. Maciej Rosoł
contact: mrosol5@gmail.com, maciej.rosol.dokt@pw.edu.pl
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
    version='v2.0.2',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/mrosol/Nonlincausality',
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
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