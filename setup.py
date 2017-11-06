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
			'psi=psi.application.cmd_launcher:main',
			'cfts=psi.application.cfts_launcher:main',
			'psi-gui=psi.application.gui_launcher:main',
		]
	},
)
