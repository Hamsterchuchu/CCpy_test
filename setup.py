import os
from setuptools import setup, find_packages

bin_path = "./CCpy/bin/"
# NOTE: os.listdir() also returns directories ('__pycache__', left behind by
# py_compile/linters) and editor backups ('*.py~'), which setuptools would then
# try to install as commands - copying a directory fails outright. Every real
# entry here is a .py file (a symlink into the package on Linux), so keep only
# those. os.path.isfile() follows symlinks, so valid links still pass.
script_files = [bin_path + f for f in os.listdir(bin_path)
                if f.endswith(".py") and os.path.isfile(os.path.join(bin_path, f))]
script_files.sort()

# -- Runtime dependencies -------------------------------------------------------
# NOTE: this list was previously built but never handed to setup(), so it had no
# effect at all; it also lost an entry to a missing comma ('matplotlib' 'pandas'
# concatenates into the nonexistent package 'matplotlibpandas').
#
# Only packages that CCpy imports on ordinary use are listed. Optional, heavy or
# currently-unused extras live in extras_require below, so a plain
# `pip install .` never drags them in.
install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'pymatgen',
    'pymatgen-analysis-diffusion',   # diffusion/AIMD analysis (split off pymatgen)
    'custodian',                     # VASP error handling (VASPio, VASPOptLoop)
    'PyYAML',                        # queue_config.yaml, vasp_default.yaml, INCAR presets
    'prettytable',                   # analyze_aimd result tables
    'ase',                           # structure engine of CCpyAlloyGen
    'spglib',                        # symmetry (also pulled in by pymatgen)
]

# Not installed by default: these back commands that are not in routine use, and
# 'tables'/'netCDF4' in particular are heavy builds that the lab environments
# have deliberately gone without.
extras_require = {
    'atk': ['tables', 'netCDF4'],                     # CCpy/ATK/*
    'extras': ['scikit-image', 'seekpath', 'mpld3'],  # 3D Brillouin zone, band paths, GaussSum plots
}

setup(
    name='CCpy',
    version='1.21',
    packages=find_packages(),
    package_data={'CCpy': ['Queue/queue_config.yaml', 'VASP/*yaml', 'VASP/vdw_kernel.bindat',
                           'SIESTA/*yaml']},
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    url='https://github.com/91bsjun/CCpy',
    license='',
    author='Byeongsun Jun',
    author_email='bjun915@gmail.com',
    description='',
    scripts=script_files
)
