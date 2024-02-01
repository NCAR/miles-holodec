from setuptools import setup, find_packages
import os

# For guidance on setuptools best practices visit
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
project_name = os.getcwd().split("/")[-1]
version = "0.1.0"
package_description = "Production version of HolodecML"
url = "https://github.com/NCAR/miles-holodec"
# Classifiers listed at https://pypi.org/classifiers/
classifiers = ["Programming Language :: Python :: 3"]
setup(name="holodec", # Change
      version=version,
      description=package_description,
      url=url,
      author="John Schreck, Matt Hayman",
      license="CC0 1.0",
      classifiers=classifiers,
      packages=find_packages(include=["holodec"]))

