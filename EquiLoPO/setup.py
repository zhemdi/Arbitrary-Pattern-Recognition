from setuptools import setup, find_packages

setup(
    name='EquiLoPO',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch', 
        'numpy',  
        'scipy', 
        'e3nn',
    ],
    author='Dmitrii Zhemchuzhnikov',
    author_email='dmitrii.zhemchuzhnikov@univ-grenoble-alpes.fr',
    description='Package EquiLoPO: Network for the equivariant recognition of arbitrary volumetric patterns in 3D.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # If your README is in Markdown
    url='https://gricad-gitlab.univ-grenoble-alpes.fr/GruLab/ILPONet/EquiLoPO',
    
)