from setuptools import setup, find_packages
from os import path


setup(name = 'pyzonotope',
    packages=find_packages(),
    version = '1.0.0',
    description = 'Zonotopes in Python',
    url = '',
    author = 'Alireza Naderi Akhormeh',
    author_email = 'ali_reza_naderi@outlook.com',
    install_requires=['numpy', 'scipy', 'cvxpy','matplotlib'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)