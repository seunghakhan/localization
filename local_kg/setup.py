from setuptools import find_packages, setup

package_name = 'local_kg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seunghakhan',
    maintainer_email='seunghakhan020306@gmail.com',
    description='Aruco detection node package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'aruco_node = local_kg.aruco_node:main',
        'imu_localization = local_kg.imu_localization:main',
        'gnss_to_enu_node = local_kg.gnss_to_enu_node:main',
        'gnss_driver_node = local_kg.gnss_driver_node:main',
    ],
},
)