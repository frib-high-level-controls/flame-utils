# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r') as f:
        return f.read()

def read_license():
    with open('LICENSE') as f:
        return f.read()

app_name = "flame_utils"
app_description = 'Utilities for FLAME Python interface'
app_long_description = readme() + '\n\n'
app_platform = ["Linux"]
app_author = "Tong Zhang"
app_author_email = "zhangt@frib.msu.edu"
app_license = read_license()
app_url = "https://github.com/frib-high-level-controls/flame-utils/"
app_keywords = "FRIB HLA high-level python FLAME online-model"
installrequires = [
    'numpy',
    'matplotlib',
    'flame-code',
]

setup(
        name=app_name,
        version="0.4.1",
        description=app_description,
        long_description=app_long_description,
        author=app_author,
        author_email=app_author_email,
        url = app_url,
        platforms=app_platform,
        license=app_license,
        keywords=app_keywords,
        packages=find_packages(exclude=['utest', 'demo', 'example']),
        classifiers=[
            'Programming Language :: Python :: 3',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering :: Physics'],
        tests_require=['nose'],
        test_suite='nose.collector',
        install_requires=installrequires,
)
