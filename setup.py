import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

setuptools.setup(
    name='natlutil',
    version='0.1.1',
    author='Alexandre H. T. Dias',
    author_email='alehenriquedias@gmail.com',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alexandredias3d/natlang',
    packages=setuptools.find_packages(),
    classifierds=[
        'Programming Language :: Python :: 3.7 ',
        'License :: OSI Approved :: Apache Software License',
    ],
    install_requires=[
        'seaborn',
        'sklearn',
        'nltk'
    ]
)
