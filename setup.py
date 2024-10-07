import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'tbsim', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# # Get the documentation
# with open(os.path.join(cwd, 'README.rst'), "r") as f:
long_description = "Starsim TB"

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
install_requires=[
    'numpy',
    'scipy',
    'pandas>=2.0.0',
    'sciris>=3.1.0',
    'matplotlib',
    'numba',
    'wheel',
    'setuptools',
    'seaborn',
    'pytest',
]
tbsim_version = version
if tbsim_version and tbsim_version < "1.0.0":
    install_requires.append("starsim==1.0.3")
else:
    install_requires.append("starsim>=2.0.0")
setup(
    name="tbsim",
    version=version,
    author="Minerva Enriquez, Robyn Stuart, Cliff Kerr, Romesh Abeysuriya, Paula Sanz-Leon, Jamie Cohen, and Daniel Klein on behalf of the tbsim Collective",
    description="tbsim",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    keywords=["agent-based model", "simulation", "disease", "epidemiology"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
)
