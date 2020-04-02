from setuptools import setup
import os.path as path


# Get the path to our current directory
path_here = path.abspath(path.dirname(__file__))

# Get the package version from its universal storage location, covid19pandas/version.py
version = {}
version_path = path.join(path_here, "covid19pandas", "version.py")
with open(version_path) as fp:
	exec(fp.read(), version)

# Get the long description from the README file
readme_path = path.join(path_here, "README.md")
with open(readme_path) as readme_file:
    readme_text = readme_file.read()

setup(name='covid19pandas',
	version=version['__version__'],
	description='COVID-19 data as pandas dataframes.',
	long_description=readme_text,
	long_description_content_type='text/markdown',
	url='https://github.com/PayneLab/covid19pandas',
	author='Dr. Samuel Payne',
	author_email='sam_payne@byu.edu',
	license='Apache 2.0',
	packages=['covid19pandas'],
	install_requires=[
		'numpy>=1.16.3',
		'pandas>=0.25.1',
		'requests>=2.21.0',
	],
    data_files=[
    ],
	classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: Apache Software License',
	],
	keywords='covid covid-19 corona coronavirus COVID COVID-19 pandas Pandas bioinformatics',
	python_requires='>=3.6.*',
	zip_safe=False,
	include_package_data=True,
	project_urls={
	   'Documentation': 'https://github.com/PayneLab/covid19pandas/tree/master/docs/'
	},
	)
