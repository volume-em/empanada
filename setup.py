#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


install_deps = [
    'torch>=1.10',
    'torchvision>=0.2',
    'numpy>=1.20',
    'numba>=0.54',
    'zarr>=2.10',
    'opencv-python>=4.5.3',
    'scikit-image>=0.18',
    'albumentations'
]

setup(
    name='empanada-dl',
    version='0.1',
    license='BSD-3',
    url='https://github.com/volume-em/empanada',
    packages=find_packages(),
    python_requires='>=3.7',
    use_scm_version=True,
    install_requires=install_deps,
    setup_requires=['setuptools_scm', 'pytest-runner'],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License'
    ],
)
