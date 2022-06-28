from setuptools import find_packages, setup
import versioneer


requirements = [
    'enaml',
    'joblib',
    'numpy',
    'palettable',
    'pyqtgraph',
    'scipy',
    'tqdm',
    'pandas',
    'pyyaml',
    'matplotlib',
]


extras_require = {
    'ni': ['pydaqmx'],
    'tdt': ['tdtpy'],
    'docs': ['sphinx', 'sphinx_rtd_theme', 'pygments-enaml'],
    'test': ['pytest', 'pytest-console-scripts'],
    'bcolz-backend': ['bcolz'],
    'legacy-bcolz-backend': ['blosc'],
    'zarr-backend': ['zarr'],
    'dev': ['coloredlogs'],
}


setup(
    name='psiexperiment',
    author='Brad Buran',
    author_email='bburan@alum.mit.edu',
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
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
            'psi-summarize-abr-gui=psi.data.io.summarize_abr:main_gui',
            'psi-summarize-abr-auto=psi.data.io.summarize_abr:main_auto',
        ]
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
