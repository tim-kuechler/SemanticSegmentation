from setuptools import setup, find_packages
import os


module_path = os.path.dirname(__file__)
setup(
    name='seg-segmentation',
    version='0.0.1',
    description='Semantic Segmantion for flickr, ..',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'ninja',
        'ml_collections',
        'albumentations',
        'opencv-python'
    ]
)
