from distutils.core import setup

setup(
    name='super-sklearn',
    version='0.1.0',
    author='Chirag Ahuja',
    description='Stacking/Super Learning Algorithm for Sklearn',
    author_email='cahuja1992@gmail.com',
    packages=['super_sklearn'],
    url='https://github.com/cahuja1992/superLearner',
    license='Apache License 2.0',
    long_description=open('README.md').read(),
    requires=[
        "numpy (>= 1.14.0)",
        "sklearn (>=0.19.2)",
        "scipy (>= 1.0.0)"
    ],
)