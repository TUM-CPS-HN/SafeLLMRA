from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ✅ Ensure launch files are installed
        (os.path.join('share', package_name, 'launch'), glob('launch/**/*.py', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='ali_reza_naderi@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "LLM_robot_controller = my_robot_controller.LLM_robot_controller:main",
            "safety_check_nonlinear_reachability= my_robot_controller.safety_check_nonlinear_reachability:main",
            "safe_LLM_controller= my_robot_controller.safe_LLM_controller:main",
            "Safety_Check_Advantage_based_Intervention= my_robot_controller.Safety_Check_Advantage_based_Intervention:main",
            "LLM_robot_controller_ABI= my_robot_controller.LLM_robot_controller_ABI:main"

        ],
    },
)
