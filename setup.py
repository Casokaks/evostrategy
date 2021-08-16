"""
Package setup
==================================

Author: Casokaks (https://github.com/Casokaks/)
Created on: Aug 15th 2021

"""


import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='projectname',
    version='0.0.1',
    author='Casokaks',
    author_email='casokaks@gmail.com',
    description='Light python template',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Casokaks/light-python-template',
    project_urls = {
        "Bug Tracker": "https://github.com/Casokaks/light-python-template/issues"
    },
    license='MIT',
    packages=['projectname'],
    install_requires=[],  # ['plotly', 'numpy', ],
)

