"""Compatibility shim — all configuration lives in pyproject.toml.

This file exists so older pip clients that lack robust PEP 660 (editable
install) support over pyproject-only projects can still do
`pip install -e .` without falling back to a build-isolated stale
setuptools. The rover_gpu container ships pip 24.x which works fine
without it, but we include this for forward-portability and to avoid
build-isolation surprises.
"""
from setuptools import setup

setup()
