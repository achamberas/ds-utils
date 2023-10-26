from setuptools import setup

setup(
    name='utils',
    version='0.1.0',    
    description='Utilities for ML pipelines',
    url='https://github.com/shuds13/pyexample',
    author='Anthony Chamberas',
    author_email='achamberas@gmail.com',
    license='BSD 2-clause',
    packages=['utils'],
    install_requires=['datetime',
                        'pytz',
                        'pandas',
                        'snowflake-connector-python',
                        'google-auth',
                        'google-auth-oauthlib ',
                        'google-cloud-core',
                        'numpy',
                        'pickleshare',
                        'matplotlib',
                        'regex',
                        'scikit-learn',
                        'scipy',                 
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)