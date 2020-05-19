import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in fishersapi/version.py
ver_file = os.path.join('fishersapi', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# read the contents of your README file into long_description
this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()    

opts = dict(name='fishersapi',
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=long_description,
            long_description_content_type='text/markdown',
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            #install_requires=REQUIRES,
            requires=REQUIRES)

install_requires = ['numpy',
                    'pandas',
                    'scipy',
                    'fisher',
                    'statsmodels'] 

if __name__ == '__main__':
    setup(**opts, install_requires = install_requires)

