# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

def read_data(metadata):
    with open('src/cudakima/cudakima.py') as file:
        for line in file.readlines():
            if '__' + metadata + '__' in line and '__main__' not in line:
                return line.split('"')[1]

setup(
    name = read_data('name'),
    version = read_data('version'),
    license = read_data('license'),
    author = read_data('author'),
    author_email = read_data('author_email'),
    package_dir={'': 'src'},
    packages = find_packages(where='src'),
    description = read_data('description'),
    install_requires=requirements,
    zip_safe=False
)