from setuptools import setup

setup(
    name='PsiExperiment',
    version='0.01',
    author='Brad Buran',
    author_email='bburan@alum.mit.edu',
    packages=['psi',],
    license='LICENSE.txt',
    description='Module for running trial-based experiments.',
	entry_points={
		'console_scripts': [
			'behavior=psi.application.base_launcher:main_animal',
			'cfts=psi.application.base_launcher:main_ear',
			'psi=psi.application.psi_launcher:main',
		]
	},
)
