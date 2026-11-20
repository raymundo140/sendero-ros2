from glob import glob

from setuptools import find_packages, setup

package_name = 'sendero'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Raymundo Gomez',
    maintainer_email='raymundo.gomez.rv@gmail.com',
    description='Supervised, Nav2-compatible navigation framework for ROS 2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigator = sendero.navigator_node:main',
        ],
    },
)
