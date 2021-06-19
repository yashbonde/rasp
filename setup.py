#!/usr/bin/python

import os
from rasp.daily import folder

from distutils.core import setup

setup(
  name='rasp',
  version='n',
  description='Restricted Access Sequence Processing (RASP) Language',
  long_description=open(os.path.join(folder(__file__), "README.md")).read(),
  author='Yash Bonde',
  author_email='bonde.yash97@gmail.com',
  url='https://github.com/yashbonde/rasp',
  packages=['rasp'],
)
