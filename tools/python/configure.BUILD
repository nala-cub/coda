# Copyright 2021 Cory Paik. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Template  for :python_configure.bzl to configure the in-build interpreter.

The in-build interpreter is only compatible with linux_x86_64. We use
platform constraints so this toolchain should only be used when it will work.

This can probably be expanded to use the @pybind11_bazel implementation of
autodection if the platform isn't linux_x86_64.
"""

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_runtime", "py_runtime_pair")

licenses(["notice"])  # PSF (cpython), BSD-3 (numpy).

package(default_visibility = ["//visibility:public"])

py_runtime(
    name = "py3_runtime",
    files = ["@org_python_cpython_38//:runtime"],
    interpreter = "@org_python_cpython_38//:bin/python",
    python_version = "PY3",
    visibility = ["//visibility:public"],
)

py_runtime_pair(
    name = "py_runtime_pair",
    py2_runtime = None,
    py3_runtime = ":py3_runtime",
)

toolchain(
    name = "py_toolchain",
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)

# alias(
#     name = "python_headers",
#     actual = "@org_python_cpython_38//:local_config_python_headers",
# )

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    includes = ["python_include"],
    deps = ["@org_python_cpython_38//:local_config_python_headers"],
)

# not sure where python_embed is used or how to validate it, but it probably
# won't work like this.
alias(
    name = "python_embed",
    actual = "@org_python_cpython_38//:local_config_python_headers",
)

cc_library(
    name = "numpy_headers",
    hdrs = [":numpy_include"],
    includes = ["numpy_include"],
)

_NUMPY_HDRS = [
    "__multiarray_api.h",
    "__ufunc_api.h",
    "_neighborhood_iterator_imp.h",
    "_numpyconfig.h",
    "arrayobject.h",
    "arrayscalars.h",
    "halffloat.h",
    "multiarray_api.txt",
    "ndarrayobject.h",
    "ndarraytypes.h",
    "noprefix.h",
    "npy_1_7_deprecated_api.h",
    "npy_3kcompat.h",
    "npy_common.h",
    "npy_cpu.h",
    "npy_endian.h",
    "npy_interrupt.h",
    "npy_math.h",
    "npy_no_deprecated_api.h",
    "npy_os.h",
    "numpyconfig.h",
    "old_defines.h",
    "oldnumeric.h",
    "random/bitgen.h",
    "ufunc_api.txt",
    "ufuncobject.h",
    "utils.h",
]

genrule(
    name = "python_include",
    srcs = ["@pip_pypi__numpy//:pkg"],
    outs = ["python_include/numpy/%s" % filename for filename in _NUMPY_HDRS],
    cmd = """
cp -r external/pip_pypi__numpy/numpy/core/include/numpy $(@D)/python_include/
""",
)

genrule(
    name = "numpy_include",
    srcs = ["@pip_pypi__numpy//:pkg"],
    outs = ["numpy_include/numpy/%s" % filename for filename in _NUMPY_HDRS],
    cmd = """
cp -r external/pip_pypi__numpy/numpy/core/include/numpy $(@D)/numpy_include/
""",
)
