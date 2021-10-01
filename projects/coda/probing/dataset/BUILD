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
load(
    "//tools:defaults.bzl",
    "py_binary",
    "py_library",
    "py_test",
    "pytype_strict_binary",
)

py_library(
    name = "_",
    srcs = ["__init__.py"],
    visibility = [":__subpackages__"],
    deps = ["//projects/coda/probing:_"],
)

py_library(
    name = "_templates",
    srcs = ["_templates.py"],
    visibility = [":__subpackages__"],
    deps = [
        ":_",
        "@dm_tree//:tree",
    ],
)

pytype_strict_binary(
    name = "dataset",
    srcs = ["dataset.py"],
    data = [
        "//projects/coda/data/coda:annotations",
        "//projects/coda/data/coda:objects",
        "//projects/coda/data/coda:templates",
    ],
    visibility = ["//projects/coda:friends"],
    deps = [
        ":_",
        ":_templates",
        "//projects/coda/probing/_src:config_util",
        "//projects/coda/probing/_src:configurable",
        "//projects/coda/probing/_src:constants",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@dm_tree//:tree",
        "@pip_pypi__datasets//:pkg",
        "@pip_pypi__gcsfs//:pkg",
        "@pip_pypi__pyclustering//:pkg",
        "@pip_pypi__pyyaml//:pkg",
        "@pip_pypi__scipy//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    deps = [
        ":dataset",
        "@absl_py//absl/testing:absltest",
    ],
)
