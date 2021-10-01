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
load("//tools:defaults.bzl", "py_library", "py_test")

package(
    default_visibility = ["//packages/labtools:internal"],
)

_DEFAULT_TEST_ENV = {
    "LABTOOLS_TESTING_INTERNAL": "true",
}

py_library(
    name = "_",
    srcs = ["__init__.py"],
    visibility = [":__subpackages__"],
    deps = ["//packages/labtools:_"],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        ":_",
        ":util",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "huggingface",
    srcs = ["huggingface.py"],
    deps = [
        ":_",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "io_util",
    srcs = ["io_util.py"],
    deps = [
        ":_",
        ":util",
        "@absl_py//absl/logging",
        "@pytoolz_cytoolz//:cytoolz",
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "profiling",
    srcs = ["profiling.py"],
    deps = [
        ":_",
        ":util",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "profiling_test",
    srcs = ["profiling_test.py"],
    env = _DEFAULT_TEST_ENV,
    deps = [
        ":_",
        ":profiling",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        ":_",
        "@absl_py//absl/logging",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    env = _DEFAULT_TEST_ENV,
    deps = [
        ":config",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@pip_pypi__ml_collections//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "io_util_test",
    srcs = ["io_util_test.py"],
    env = _DEFAULT_TEST_ENV,
    deps = [
        ":io_util",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
