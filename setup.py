from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='angorapy',
    version='0.8.1',
    description='ANthropomorphic Goal-ORiented Modeling, Learning and Analysis for Neuroscience',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/ccnmaastricht/dexterous-robot-hand',
    author='Tonio Weidler',
    author_email='research@tonioweidler.de',
    license='GPL-3.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.23.0",
        "Box2D",
        "gym==0.24.0",
        "mujoco",
        "tensorflow==2.10.0",
        "mpi4py==3.1.3",
        "tqdm",
        "simplejson",
        "psutil",
        "scipy",
        "scikit-learn",
        "argcomplete",
        "matplotlib",
        "scikit-learn==0.24.1",
        "pandas==1.4.4",
        "nvidia-ml-py3",
        "seaborn",
        "distance",
        "protobuf==3.19.0",
        "panda_gym",

        # webinterface
        "itsdangerous==2.0.1",
        "werkzeug==2.0.3",
        "Flask~=1.1.2",
        "Jinja2==3.0.0",
        "bokeh",
        "flask_jsglue",
    ],

    package_data={
        "angorapy": ["environments/assets/**/*"],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
