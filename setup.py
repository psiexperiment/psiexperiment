from distutils.core import setup

setup(
    name='PsiExperiment',
    version='0.01',
    author='Brad Buran',
    author_email='bburan@alum.mit.edu',
    packages=[
        'psi',
        'experiments',
    ],
    license='LICENSE.txt',
    description='Module for running trial-based experiments.',
)
