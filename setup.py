import os
import sys
import shutil
import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension


def clean_repo():
    repo_folder = os.path.realpath(os.path.dirname(__file__))
    dist_folder = os.path.join(repo_folder, 'dist')
    build_folder = os.path.join(repo_folder, 'build')
    if os.path.isdir(dist_folder):
        shutil.rmtree(dist_folder, ignore_errors=True)
    if os.path.isdir(build_folder):
        shutil.rmtree(build_folder, ignore_errors=True)


# Read version string
_version = None
script_folder = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(script_folder, 'ibug', 'face_pose_augmentation', '__init__.py')) as init:
    for line in init.read().splitlines():
        fields = line.replace('=', ' ').replace('\'', ' ').replace('\"', ' ').replace('\t', ' ').split()
        if len(fields) >= 2 and fields[0] == '__version__':
            _version = fields[1]
            break
if _version is None:
    sys.exit('Sorry, cannot find version information.')

# Configure cython modules
package_location = ['ibug', 'face_pose_augmentation', 'fpa']
extensions = [
    Extension('.'.join(package_location + ['pyMM3D']),
              sources=[os.path.join(*(package_location + ['pyMM3D.pyx'])),
                       os.path.join(*(package_location + ['cpp', 'MM3D.cpp']))],
              include_dirs=[np.get_include()],
              language='c++'),
    Extension('.'.join(package_location + ['pyFaceFrontalization']),
              sources=[os.path.join(*(package_location + ['pyFaceFrontalization.pyx'])),
                       os.path.join(*(package_location + ['cpp', 'MM3D.cpp'])),
                       os.path.join(*(package_location + ['cpp', 'face_frontalization.cpp']))],
              include_dirs=[np.get_include()],
              language='c++')]

# Installation
config = {
    'name': 'ibug_face_pose_augmentation',
    'version': _version,
    'description': 'A face pose augmentation toolbox based on 3DDFA.',
    'author': 'Jie Shen',
    'author_email': 'js1907@imperial.ac.uk',
    'packages': ['ibug.face_pose_augmentation'],
    'install_requires': ['numpy>=1.16.0', 'scipy>=1.1.0', 'torch>=1.1.0', 'opencv-python>=3.4.2',
                         'cmake>=3.16', 'igraph>=0.8.3', 'matplotlib', 'shapely', 'cython'],
    'ext_modules': cythonize(extensions, compiler_directives={'language_level': '3'}),
    'zip_safe': False
}
clean_repo()
setup(**config)
clean_repo()
