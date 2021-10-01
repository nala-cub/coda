# optax BUILD
#   For: https://github.com/deepmind/optax
#

load(
    "@com_github_corypaik_coda//tools:defaults.bzl",
    "py_generate_test_suite",
)
load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "optax",
    srcs = glob(
        ["optax/**/*.py"],
        exclude = [
            "**/test_*.py",
            "**/*_test.py",
        ],
    ),
    imports = ["."],
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl/logging",
        "@com_google_jax//jax",
        "@dm_chex//:chex",
        "@pip_pypi__numpy//:pkg",
    ],
)

py_generate_test_suite(
    name = "tests",
    srcs = glob([
        "optax/**/test_*.py",
        "optax/**/*_test.py",
    ]),
    env = {
        "CUDA_VISIBLE_DEVICES": "",
    },
    deps = [
        ":optax",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@com_google_flax//:flax",
        "@pip_pypi__dm_haiku//:pkg",
        "@pip_pypi__tabulate//:pkg",
    ],
)
