"""
evostrategy setup
==================================
Python library implementing Machine Learning Evolution Strategy.

Author: Casokaks (https://github.com/Casokaks/)
Created on: Aug 15th 2021

"""

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from setuptools import setup, find_packages
setup(
    name='evostrategy',
    version='0.2.1',
    author='Casokaks',
    author_email='casokaks@gmail.com',
    description='Python library implementing Machine Learning Evolution Strategy.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Casokaks/evostrategy',
    project_urls = {
        "Bug Tracker": "https://github.com/Casokaks/evostrategy/issues"
    },
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  
)

