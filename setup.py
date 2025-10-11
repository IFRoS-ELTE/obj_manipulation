# setup.py
from setuptools import setup, find_packages

package_name = 'obj_manipulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, '{}.*'.format(package_name)]),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='',
    maintainer_email='you@example.com',
    description='Object manipulation ROS1 package',
    license='MIT',
)
