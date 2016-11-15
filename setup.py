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
			'psi = psi.application.__main__:main'
		]
	},
)
