import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'particle_filter'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        # Install marker file
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Package XML
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Install Python modules
        (os.path.join('lib', package_name), glob('src/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Hariharan Ravichandran',
    author_email='hariharanr1799@gmail.com',
    maintainer='Hariharan Ravichandran',
    maintainer_email='hariharanr1799@gmail.com',
    description='ROS 2 Version of Particle Filter Localization using RangeLibc for accelerated ray casting.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'synPF = synPF:main',
        ],
    },
)
