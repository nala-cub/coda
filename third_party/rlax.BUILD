# rlax BUILD
#   For: https://github.com/deepmind/rlax
#

load(
    "@com_github_corypaik_coda//tools:defaults.bzl",
    "py_generate_test_suite",
    "py_library",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "rlax",
    srcs = glob(
        ["rlax/**/*.py"],
        exclude = [
            "**/*_test.py",
        ],
    ),
    imports = ["."],
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl/logging",
        "@com_google_jax//jax",
        "@dm_chex//:chex",
        "@pip_pypi__dm_env//:pkg",
        "@pip_pypi__numpy//:pkg",
    ],
)

py_generate_test_suite(
    name = "tests",
    srcs = glob([
        "rlax/**/*_test.py",
    ]),
    env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    },
    deps = [
        ":rlax",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@dm_optax//:optax",
        "@pip_pypi__dm_haiku//:pkg",
        "@pip_pypi__tabulate//:pkg",
    ],
)
