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
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

py_binary(
    name = "run_probing",
    srcs = ["run_probing.py"],
    deps = [
        "//packages/labtools",
        "//projects/coda/probing/_src:config_util",
        "//projects/coda/probing/_src:configurable",
        "//projects/coda/probing/_src:constants",
        "//projects/coda/probing/_src:fwd_steps",
        "//projects/coda/probing/_src:preprocess_steps",
        "//projects/coda/probing/dataset",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@pip_pypi__datasets//:pkg",
        "@pip_pypi__icecream//:pkg",
        "@pip_pypi__torch//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_binary(
    name = "run_scoring",
    srcs = ["run_scoring.py"],
    deps = [
        "//packages/labtools",
        "//projects/coda/probing/_src:configurable",
        "//projects/coda/probing/_src:constants",
        "//projects/coda/probing/_src:metrics",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@pip_pypi__pandas//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_binary(
    name = "zeroshot",
    srcs = ["zeroshot.py"],
    args = [
        "--zeroshot.ngram_gbc_path",
        "$(location //projects/coda/data:ngram-counts-gbc.csv)",
    ],
    data = [
        "//projects/coda/data:ngram-counts-gbc.csv",
    ],
    visibility = ["//projects/coda:friends"],
    deps = [
        ":run_probing",
        ":run_scoring",
        "//packages/labtools",
        "//projects/coda/probing:configs",
        "//projects/coda/probing/_src:configurable",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@pip_pypi__gcsfs//:pkg",
        "@pip_pypi__ml_collections//:pkg",
        "@pip_pypi__pandas//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "zeroshot_test",
    size = "enormous",
    srcs = ["zeroshot_test.py"],
    deps = [":zeroshot"],
)
