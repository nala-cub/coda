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
    visibility = [":__subpackages__"],
    deps = ["//projects/coda/probing:_"],
)

py_binary(
    name = "build_reprs",
    srcs = ["build_reprs.py"],
    deps = [
        "//packages/labtools",
        "//projects/coda/probing/_src:config_util",
        "//projects/coda/probing/_src:configurable",
        "//projects/coda/probing/_src:fwd_steps",
        "//projects/coda/probing/_src:preprocess_steps",
        "//projects/coda/probing/dataset",
        "@absl_py//absl:app",
        "@absl_py//absl/logging",
        "@com_github_openai_clip//:clip",
        "@hf_transformers",
        "@pip_pypi__datasets//:pkg",
        "@pip_pypi__torch//:pkg",
        "@pip_pypi__tqdm//:pkg",
    ],
)

py_binary(
    name = "repr_probing",
    srcs = ["repr_probing.py"],
    deps = [
        ":_",
        ":data",
        ":models",
        "//projects/coda/probing/_src:configurable",
        "//projects/coda/probing/_src:constants",
        "@absl_py//absl:app",
        "@absl_py//absl/logging",
        "@com_google_jax//jax",
        "@dm_chex//:chex",
        "@dm_tree//:tree",
        "@pip_pypi__datasets//:pkg",
        "@pip_pypi__einops//:pkg",
        "@pip_pypi__icecream//:pkg",
        "@pip_pypi__numpy//:pkg",
        "@pip_pypi__pandas//:pkg",
        "@pip_pypi__tqdm//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "repr_probing_test",
    size = "small",
    srcs = ["repr_probing_test.py"],
    deps = [
        ":repr_probing",
        "//projects/coda/probing:configs",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        ":_",
        "@com_google_jax//jax",
        "@dm_chex//:chex",
        "@pip_pypi__datasets//:pkg",
        "@pip_pypi__dm_haiku//:pkg",
        "@pip_pypi__numpy//:pkg",
    ],
)

py_test(
    name = "data_test",
    size = "small",
    srcs = ["data_test.py"],
    deps = [
        ":data",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        "@com_google_jax//jax",
        "@dm_optax//:optax",
        "@dm_rlax//:rlax",
        "@pip_pypi__dm_haiku//:pkg",
        "@pip_pypi__icecream//:pkg",
        "@pip_pypi__numpy//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":data",
        ":models",
        "//packages/labtools",
        "//projects/coda/probing/_src:metrics",
        "@com_google_jax//jax",
        "@dm_chex//:chex",
        "@pip_pypi__absl_py//:pkg",
        "@pip_pypi__dm_haiku//:pkg",
    ],
)

py_binary(
    name = "repr_scoring",
    srcs = ["repr_scoring.py"],
    visibility = ["//probing:__pkg__"],
    deps = [
        "//projects/coda/probing/_src:configurable",
        "//projects/coda/probing/_src:constants",
        "//projects/coda/probing/_src:metrics",
        "@absl_py//absl:app",
        "@absl_py//absl/logging",
        "@pip_pypi__numpy//:pkg",
        "@pip_pypi__pandas//:pkg",
    ],
)

####
# pipeline
py_binary(
    name = "representations",
    srcs = ["representations.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_",
        ":build_reprs",
        ":repr_probing",
        ":repr_scoring",
        "//packages/labtools",
        "//projects/coda/probing:configs",
        "//projects/coda/probing/dataset",
        "@absl_py//absl:app",
        "@absl_py//absl/logging",
        "@dm_chex//:chex",
        "@pip_pypi__gcsfs//:pkg",
        "@pip_pypi__ml_collections//:pkg",
        "@pip_pypi__pandas//:pkg",
        "@pytoolz_cytoolz//:cytoolz",
    ],
)

py_test(
    name = "representations_test",
    size = "enormous",
    srcs = ["representations_test.py"],
    deps = [
        ":representations",
        "//projects/coda/probing:configs",
        "@absl_py//absl/testing:absltest",
    ],
)
