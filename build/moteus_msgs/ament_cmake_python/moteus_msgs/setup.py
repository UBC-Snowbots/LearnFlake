from setuptools import find_packages
from setuptools import setup

setup(
    name='moteus_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('moteus_msgs', 'moteus_msgs.*')),
)
