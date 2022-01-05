from setuptools import setup
from setuptools import find_packages
import os 
import subprocess
import re 
import io 

PACKAGE_NAME = 'regtricks'
ROOTDIR = os.path.abspath(os.path.dirname(__file__))


def get_filetext(filename):
    """ Get the text of a local file """
    with io.open(os.path.join(ROOTDIR, filename), encoding='utf-8') as f:
        return f.read()


def git_version():
    """ Get the full and python standardized version from Git tags (if possible) """
    try:
        # Full version includes the Git commit hash
        full_version = subprocess.check_output('git describe', shell=True).decode("utf-8").strip(" \n")

        # Python standardized version in form major.minor.patch.post<build>
        version_regex = re.compile(r"v?(\d+\.\d+\.\d+(-\d+)?).*")
        match = version_regex.match(full_version)
        if match:
            std_version = match.group(1).replace("-", ".post")
        else:
            raise RuntimeError("Failed to parse version string %s" % full_version)
        return full_version, std_version

    except Exception:
        # Any failure, return None. We may not be in a Git repo at all
        return None, None


def git_timestamp():
    """ Get the last commit timestamp from Git (if possible)"""
    try:
        return subprocess.check_output('git log -1 --format=%cd', shell=True).decode("utf-8").strip(" \n")
    except Exception:
        # Any failure, return None. We may not be in a Git repo at all
        return None


def update_metadata(version_str, timestamp_str):
    """ Update the version and timestamp metadata in the module _version.py file """
    with io.open(os.path.join(ROOTDIR, PACKAGE_NAME, "_version.py"), "w", encoding='utf-8') as f:
        f.write("__version__ = '%s'\n" % version_str)
        f.write("__timestamp__ = '%s'\n" % timestamp_str)


def get_requirements():
    """ Get a list of all entries in the requirements file """
    with io.open(os.path.join(ROOTDIR, 'requirements.txt'), encoding='utf-8') as f:
        # print([l.strip() for l in f.readlines()])
        return [l.strip() for l in f.readlines()]


def get_version():
    """ Get the current version number (and update it in the module _version.py file if necessary)"""
    version, timestamp = git_version()[1], git_timestamp()

    if version is not None and timestamp is not None:
        # We got the metadata from Git - update the version file
        update_metadata(version, timestamp)
    else:
        # Could not get metadata from Git - use the version file if it exists
        try:
            with io.open(os.path.join(ROOTDIR, PACKAGE_NAME, '_version.py'), encoding='utf-8') as f:
                md = f.read()
                match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", md, re.M)
                if match:
                    version = match.group(1)
                else:
                    raise ValueError("Stored version could not be parsed")
        except (IOError, ValueError):
            version = "unknown"
            update_metadata(version, "unknown")
    return version

if __name__ == "__main__":
    
    setup(name=PACKAGE_NAME,
        version=get_version(),
        description="Tools for manipulating and applying registrations",
        long_description=get_filetext('README.md'),
        long_description_content_type='text/markdown',
        author='Tom Kirk',
        author_email='thomas.kirk@eng.ox.ac.uk',
        license='BSD-3-clause', 
        url='https://github.com/tomfrankkirk/regtricks',
        install_requires=get_requirements(),
        packages=find_packages())