"""Example script to create a exe of the breakout example using py2exe.

Tested on py2exe 0.9.2.2 on python 3.4
"""

import os
from distutils.core import setup

import py2exe

import easymunk

setup(
    windows=["breakout.py"],
    data_files=[("", [easymunk.chipmunk_path])],
    zipfile=None,
    options={"py2exe": {"bundle_files": 1}},
)
