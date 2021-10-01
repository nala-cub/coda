# chex BUILD
#   For: https://github.com/deepmind/chex
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
    name = "chex",
    srcs = glob(
        ["chex/**/*.py"],
        exclude = [
            "**/*_test.py",
        ],
    ),
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl/logging",
        "@com_google_jax//jax",
        "@dm_tree//:tree",
        "@pip_pypi__numpy//:pkg",
        "@pytoolz_toolz//:toolz",
    ],
)

# currently we are running this on cpu only, because for all tests to pass we
# need multiple devices. this is also much simpler for ci. realisticlly I'll
# probably just run a buildkite worker from time to time on my machine with gpus
# and then 90% of the time test on cpu.
# TODO(corypaik): Add asserts_internal_test back to the  test suite once error
# is resolved. Message:
# ----------------------------------------------------------------------
# Traceback (most recent call last):
#  ...
#  in bound_param_test
#     return test_method(self, *testcase_params)
#   .../dm_chex/chex/_src/asserts_internal_test.py", line 34, in test_is_traceable
#     prev_state = jax.api.FLAGS.experimental_cpp_jit
# AttributeError: module 'jax' has no attribute 'api'
py_generate_test_suite(
    name = "tests",
    srcs = glob(
        [
            "chex/**/*_test.py",
        ],
        exclude = [
            "chex/_src/asserts_internal_test.py",
        ],
    ),
    env = {
        "CUDA_VISIBLE_DEVICES": "",
    },
    deps = [
        ":chex",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
