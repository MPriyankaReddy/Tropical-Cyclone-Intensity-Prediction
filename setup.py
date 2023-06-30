from setuptools import setup, find_packages

setup(
    name='packageC45',
    version='0.1.0',
    packages=find_packages(include=['packageC45', 'packageC45.*'])
)