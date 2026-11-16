"""Compile arch's Cython recursions into this checkout (Artemis setup command).

    $VENV/bin/python -I -B artemis_bench/build_ext.py

Same Cython directives as arch's meson build (arch/meson.build), so the module
is the one a pip install builds. Also writes arch/_version.py, which a meson
build would generate. Both outputs are gitignored. Build files go to a
temporary folder, never into the repository.
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from Cython.Build import cythonize  # noqa: E402
from setuptools import Extension, setup  # noqa: E402

DIRECTIVES = dict(language_level=3, cpow=True, boundscheck=False, wraparound=False,
                  cdivision=True, binding=True)

with open(os.path.join(ROOT, "arch", "_version.py"), "w") as fh:
    fh.write('version = "8.0.0.dev0"\nversion_tuple = (8, 0, 0, "dev0")\n'
             "__version__ = version\n__version_tuple__ = version_tuple\n")

tmp = tempfile.mkdtemp(prefix="arch-build-")
try:
    ext = Extension("arch.univariate.recursions", ["arch/univariate/recursions.pyx"],
                    include_dirs=[np.get_include()],
                    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                    extra_compile_args=["-O2"])
    mods = cythonize([ext], quiet=True, compiler_directives=DIRECTIVES, build_dir=tmp)
    setup(script_args=["-q", "build_ext", "--inplace", "--build-temp", tmp, "--force"],
          ext_modules=mods, name="arch-recursions", packages=[], py_modules=[])
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("build_ext: arch.univariate.recursions compiled")
