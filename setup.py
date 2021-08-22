# Standard imports
import sys
import glob, os
from setuptools import setup, find_packages


# Begin setup
setup_keywords = dict()
setup_keywords['name'] = 'rfmhw'
setup_keywords['description'] = 'Random Forest analysis for Marine Heat Wave events'
setup_keywords['author'] = 'Kat Giamalaki et al.'
setup_keywords['author_email'] = 'agiamala@ucsc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/KatGiamalaki/Random_Forest_for_MHW'
setup_keywords['version'] = '0.0.dev0'
# Use README.rst as long_description.
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>3.7.0)']
setup_keywords['install_requires'] = [
    'xarray', 'seaborn', 'scikit-learn', 'scikit-image'] 
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages()
setup_keywords['setup_requires'] = ['pytest-runner']
setup_keywords['tests_require'] = ['pytest']

if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if not os.path.basename(fname).endswith('.rst')]

setup(**setup_keywords)