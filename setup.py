from setuptools import setup, find_packages

exec(open('equiformer_pytorch/version.py').read())

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
  include_package_data = True,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
