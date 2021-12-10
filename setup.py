from setuptools import setup, find_packages

import os

version_file=open(os.path.join("./ALS", 'VERSION'))
version = version_file.read().strip()

setup(
    name='ALSCPDpip',
    version=version,
    description='ALSCPD source files',
    author='Christian Delavier',
    author_email='delavier@stud.uni-heidelberg.de',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'tensorly'],
)