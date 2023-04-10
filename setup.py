from setuptools import setup, find_packages
from samclipdiffusion import __version__

setup(
    name='samclipdiffusion',
    version='0.0.1b',
    description='A package for clip-guided diffusion',
    author='Eren Akbulut',
    author_email='erenakbulutwork@gmail.com',
    url='https://github.com/eren23/sam-clip-diffusion',
    packages=['samclipdiffusion', 'samclipdiffusion.guided_diffusion_pipeline'],
    install_requires=[
        'torch',
        'opencv-python',
        'Pillow',
        'git+https://github.com/openai/CLIP.git',
        'git+https://github.com/facebookresearch/segment-anything.git',
        'diffusers',
        'transformers',
        'accelerate'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
