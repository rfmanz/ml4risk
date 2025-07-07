import os
from setuptools import setup, find_packages

with open(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "README.md"
    )
) as f:
    readme = f.read()

try:  # automatically updates when pushed to gitlab
    version = '0.1.' + str(int(os.environ['CI_PIPELINE_IID'])) 
except:
    version = '0.1.1'  # for local tox operations
     
package_name = 'ml4risk'

requirements = ['black',
 'boto3>=1.9.188',
 'botocore>=1.12.188',
 'matplotlib>=3.0.3',
 'numpy>=1.16.5',
 'pandas-profiling>=2.8.0',
 'pandas>=0.24.2',
 'pyarrow>=0.14.1',
 'pytest==6.2.4',
 's3fs>=0.3.1',
 'scikit-learn>=0.20.3',
 'seaborn>=0.9.0',
 'tox>=3.23.1',
 'tqdm>=4.32.2',
 'twine>=1.13.0',
 'smart-open']


setup(
    name=package_name,
    version=version,
    author="SoFi",
    author_email="jxu@sofi.org",
    description="Risk data science ML4Risk Framework",
    packages=find_packages(),
    license="proprietary",
    long_description=readme,
    keywords="machine-learning",
    entry_points={
        'console_scripts': [
        ]
    },
    install_requires=requirements,
    extras_require={
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: Proprietary :: All Rights Reserved by SoFi Inc.",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Operating System :: POSIX"
    ]
)
