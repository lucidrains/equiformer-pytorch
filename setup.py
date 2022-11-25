from setuptools import setup, find_packages

import sys
from functools import lru_cache
from subprocess import DEVNULL, call
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

exec(open('equiformer_pytorch/version.py').read())

@lru_cache(None)
def cuda_toolkit_available():
  try:
    call(["nvcc"], stdout = DEVNULL, stderr = DEVNULL)
    return True
  except FileNotFoundError:
    return False

def compile_args():
  args = ["-fopenmp", "-ffast-math"]
  if sys.platform == "darwin":
    args = ["-Xpreprocessor", *args]
  return args

def ext_modules():
  if not cuda_toolkit_available():
    return []

  return [
    CUDAExtension(
      __cuda_pkg_name__,
      sources = ["equiformer_pytorch/equiformer_pytorch.cu"]
    )
  ]

setup(
  name = 'equiformer-pytorch',
  packages = find_packages(exclude=[]),
  version = __version__,
  license='MIT',
  description = 'Equiformer - SE3/E3 Graph Attention Transformer for Molecules and Proteins',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/equiformer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'equivariance',
    'molecules',
    'proteins'
  ],
  install_requires=[
    'beartype',
    'einops>=0.6',
    'filelock',
    'numpy',
    'torch>=1.6',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'lie_learn',
    'numpy',
    'pytest'
  ],
  ext_modules = ext_modules(),
  cmdclass = {"build_ext": BuildExtension},
  include_package_data = True,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
