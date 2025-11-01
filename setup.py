#!/usr/bin/python

import os
from setuptools import setup

setup(
  name='rasp',
  version='0.1.0',
  description='Restricted Access Sequence Processing (RASP) Language',
  long_description=open("README.md").read(),
  author='Yash Bonde',
  author_email='bonde.yash97@gmail.com',
  url='https://github.com/yashbonde/rasp',
  packages=['rasp'],
  install_requires=[
    'numpy',
    'tqdm',
    'torch',
    'einops',
    'requests'
  ],
)
