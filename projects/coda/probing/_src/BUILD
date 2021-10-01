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
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "//projects/coda/probing:__subpackages__",
])

py_library(
    name = "_",
    srcs = ["__init__.py"],
    visibility = [":__subpackages__"],
    deps = ["//projects/coda/probing:_"],
)

py_library(
    name = "fwd_steps",
    srcs = ["fwd_steps.py"],
    deps = [
        ":_",
        "//packages/labtools",
        "@dm_tree//:tree",
        "@pip_pypi__einops//:pkg",
        "@pip_pypi__torch//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "fwd_steps_test",
    size = "small",
    srcs = ["fwd_steps_test.py"],
    deps = [
        ":fwd_steps",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "config_util",
    srcs = ["config_util.py"],
    deps = [
        ":_",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@hf_transformers",
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        ":_",
        "@pip_pypi__scipy//:pkg",
    ],
)

py_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@com_google_jax//jax",
        "@pip_pypi__numpy//:pkg",
        "@pip_pypi__torch//:pkg",
    ],
)

py_library(
    name = "preprocess_steps",
    srcs = ["preprocess_steps.py"],
    deps = [
        ":constants",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@hf_transformers",
        "@pip_pypi__datasets//:pkg",
    ],
)

py_test(
    name = "preprocess_steps_test",
    srcs = ["preprocess_steps_test.py"],
    deps = [
        ":preprocess_steps",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@hf_transformers",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [":_"],
)

py_library(
    name = "configurable",
    srcs = ["configurable.py"],
    deps = [
        ":_",
        "@absl_py//absl/flags",
        "@dm_fancyflags//:fancyflags",
    ],
)

py_test(
    name = "configurable_test",
    size = "small",
    srcs = ["configurable_test.py"],
    deps = [
        ":configurable",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "result_util",
    srcs = ["result_util.py"],
    deps = [
        ":constants",
        "@absl_py//absl/logging",
        "@pip_pypi__numpy//:pkg",
        "@pip_pypi__pandas//:pkg",
    ],
)
