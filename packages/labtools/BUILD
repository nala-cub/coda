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
load("@rules_python//python:defs.bzl", "py_library")
load("//tools:defaults.bzl", "pytype_strict_library")

package_group(
    name = "internal",
    packages = [
        "//packages/labtools/...",
    ],
)

# Sepearte the stub so we can use it during testing without depending on
# the labtools package.
py_library(
    name = "_",
    srcs = ["__init__.py"],
    imports = [".."],
    visibility = [":__subpackages__"],
)

pytype_strict_library(
    name = "labtools",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_",
        "//packages/labtools/_src:config",
        "//packages/labtools/_src:huggingface",
        "//packages/labtools/_src:io_util",
        "//packages/labtools/_src:profiling",
        "//packages/labtools/_src:util",
    ],
)
