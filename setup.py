from setuptools import setup, find_packages


from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

packages = find_packages()
packages.append('scripts')

setup(
    name='pl_hsir',
    description="Hyperspectral Image Restoration Tools",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=packages,
    package_dir={'hst_engine': 'hst_engine', 'scripts': 'scripts'},
    version='0.0.1',
    include_package_data=True,
    install_requires=['tensorboard', 'torch', 'torchvision', 'numpy', 'lightning', 'timm', 'einops', 'scipy'],
)