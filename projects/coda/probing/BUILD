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

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "_",
    srcs = ["__init__.py"],
    imports = [".."],
    visibility = [":__subpackages__"],
)

py_library(
    name = "probing",
    visibility = ["//projects/coda:friends"],
    deps = [
        ":_",
        ":configs",
        ":ngram_stats",
        "//projects/coda/probing/representations",
        "//projects/coda/probing/zeroshot",
    ],
)

py_binary(
    name = "ngram_stats",
    srcs = ["ngram_stats.py"],
    args = [
        "--ngrams.ngram_gbc_path",
        "$(location //projects/coda/data:ngram-counts-gbc.csv)",
    ],
    data = ["//projects/coda/data:ngram-counts-gbc.csv"],
    visibility = ["//projects/coda:friends"],
    deps = [
        "//projects/coda/probing/_src:configurable",
        "//projects/coda/probing/_src:constants",
        "//projects/coda/probing/_src:metrics",
        "//projects/coda/probing/dataset",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@pip_pypi__fsspec//:pkg",
        "@pip_pypi__gcsfs//:pkg",
        "@pip_pypi__numpy//:pkg",
        "@pip_pypi__pandas//:pkg",
    ],
)

py_library(
    name = "configs",
    srcs = ["configs.py"],
    visibility = ["//projects/coda:internal"],
    deps = [
        ":_",
        "//packages/labtools",
        "@absl_py//absl/logging",
        "@pip_pypi__ml_collections//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "configs_test",
    size = "small",
    srcs = ["configs_test.py"],
    srcs_version = "PY3",
    deps = [
        ":configs",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@dm_tree//:tree",
        "@pip_pypi__absl_py//:pkg",
    ],
)
