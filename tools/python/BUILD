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
""" configure a python interpreter compatible with pip_install """

load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "pytype_helper",
    srcs = ["pytype_helper.py"],
    data = ["@python_typeshed//:files"],
    visibility = ["//visibility:public"],
    deps = ["@pip_pypi__pytype//:pkg"],
)

py_library(
    name = "py_repl",
    srcs = ["py_repl.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "jupyterlab_helper",
    srcs = ["jupyterlab_helper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip_pypi__ipywidgets//:pkg",
        "@pip_pypi__jupyter_contrib_nbextensions//:pkg",
        "@pip_pypi__jupyterlab//:pkg",
    ],
)

exports_files([
    "configure.bzl",
    "interpreter.bzl",
])
