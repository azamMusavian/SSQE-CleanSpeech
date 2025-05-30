from setuptools import setup, find_packages

setup(
    name="vqscore",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'soundfile',
        'numpy',
        'logging',
        'pandas',
        'matplotlib',
        'torchaudio',
        'tensorboardX',
        'tqdm',
        'librosa',
        'pyyaml',

    ],
)