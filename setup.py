# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:22:58 2023

@author: ZHANG Jun
"""

from distutils.core import setup
from setuptools import find_packages


with open("README.md", "r", encoding='utf-8') as f:
  long_description = f.read()

# with open('requirements.txt', "r") as f:
#     requirements = f.readlines()
#     requirements = [x.strip() for x in requirements]

setup(name='elasticnet',
      version='1.0.4',
      python_requires='>=3.9',
      description='Predict the mechanical properties of multi-component transition metal carbides (MTMCs).',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='ZHANG Jun; ZHAO Shijun',
      author_email='j.zhang@my.cityu.edu.hk',
      url='https://github.com/jzhang-github/MTMC_elastic_modulus',
      install_requires=['numpy>=1.25.0',
 'scikit-learn==1.2.2',
 'tensorflow==2.10.0',
 'ase==3.22.1',
 'pandas==1.5.3',
 'openpyxl==3.0'],
      license='MIT',
      packages=find_packages(),
      platforms=["all"],
      classifiers=[
                # How mature is this project? Common values are
                #   3 - Alpha
                #   4 - Beta
                #   5 - Production/Stable
                'Development Status :: 3 - Alpha',
                'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.10',
          'Topic :: Software Development :: Libraries'
      ],
      )
