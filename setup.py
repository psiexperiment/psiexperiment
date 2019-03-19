from setuptools import find_packages, setup

requirements = [
    'json_tricks',
    'palettable',
    'pydaqmx',
]

package_data = {
    '': ['*.enaml', '*.txt'],
}


setup(
    name='PsiExperiment',
    version='0.01',
    author='Brad Buran',
    author_email='bburan@alum.mit.edu',
    install_requires=requirements,
    packages=find_packages(),
    package_data=package_data,
    license='LICENSE.txt',
    description='Module for running trial-based experiments.',
    entry_points={
        'console_scripts': [
            'psi=psi.application.psi_launcher:main',
            'psi-config=psi.application:config',
            'psi-behavior=psi.application.base_launcher:main_animal',
            'psi-calibration=psi.application.base_launcher:main_calibration',
            'psi-cfts=psi.application.base_launcher:main_ear',
            'psi-cohort=psi.application.base_launcher:main_cohort',
            'psi-summarize-abr=psi.data.io.summarize_abr:main',
            'psi-summarize-abr-auto=psi.data.io.summarize_abr:main_auto',
        ]
    },
)
