"""Test package for llamagui.

The tests import the package by absolute name, so the parent directory is added
to ``sys.path`` from within the test module itself.  That keeps
``python -m unittest discover -s tests`` working from the ``gui`` directory with
no installation step and no ``conftest``-style magic.
"""
