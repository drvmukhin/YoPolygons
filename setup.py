from setuptools import setup, find_packages

setup(
    name='YoPolygons',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy'
    ],
    author='Vasily Mukhin',
    author_email='vmukhin.dev@gmail.com',
    description='A set of utilities to manipulate with images',
    url='https://github.com/drvmukhin/YoPolygons.git',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    python_requires='>=3.10',
)