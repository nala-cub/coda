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
""" Explicit exports of default rules. """

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    _pybind_extension = "pybind_extension",
)
load(
    "@org_tensorflow//tensorflow/core/platform/default:build_config.bzl",
    _pyx_library = "pyx_library",
)
load(
    "//tools/python:defs.bzl",
    _jupyterlab_server = "jupyterlab_server",
    _py_binary = "py_binary",
    _py_generate_test_suite = "py_generate_test_suite",
    _py_library = "py_library",
    _py_repl = "py_repl",
    _py_test = "py_test",
    _pytype_binary = "pytype_binary",
    _pytype_library = "pytype_library",
    _pytype_strict_binary = "pytype_strict_binary",
    _pytype_strict_library = "pytype_strict_library",
)

py_repl = _py_repl

py_binary = _py_binary
py_library = _py_library
py_test = _py_test

pytype_binary = _pytype_binary
pytype_library = _pytype_library
pytype_strict_binary = _pytype_strict_binary
pytype_strict_library = _pytype_strict_library

py_generate_test_suite = _py_generate_test_suite

pyx_library = _pyx_library
pybind_extension = _pybind_extension

jupyterlab_server = _jupyterlab_server
