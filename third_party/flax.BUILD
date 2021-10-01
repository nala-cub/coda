# Description:
#  Flax is a high-performance neural network library and ecosystem for JAX that
#  is designed for flexibility: Try new forms of training by forking an example
#  and by modifying the training loop, not by adding features to a framework.

load(
    "@com_github_corypaik_coda//tools:defaults.bzl",
    "py_generate_test_suite",
)
load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])  # Apache-2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "flax",
    srcs = glob(["flax/**/*.py"]),
    imports = ["."],
    srcs_version = "PY3",
    deps = [
        "@com_google_jax//jax",
        "@dm_optax//:optax",
        "@pip_pypi__numpy//:pkg",
        "@pip_pypi__msgpack//:pkg",
        # matplotlib",  # only needed for tensorboard export
    ],
)

# Disable pre-allocation during tests, otherwise we'd have to run them all
# sequentially. Most tests use little GPU memeory, so fragmentation isn't much
# of a concern here.
py_generate_test_suite(
    name = "tests",
    srcs = glob(
        ["tests/**/*.py"],
        exclude = [
            "tests/tensorboard_test.py",  # requires tensorboard
            "tests/checkpoints_test.py",  # tensorflow
            # TODO(corypaik): broken tests
            "tests/linen_transforms_test.py",
            "tests/optim_test.py",
        ],
    ),
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    deps = [
        ":flax",
        "@com_google_jax//jax",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        # "atari-py==0.2.5",  # Last version does not have the ROMs we test on pre-packaged
        # "clu",  # All examples.
        # "gym==0.18.3",
        # "ml-collections",
        # "opencv-python",
        # "pytest",
        # "pytest-cov",
        # "pytest-xdist==1.34.0",  # upgrading to 2.0 broke tests, need to investigate
        # "pytype==2021.5.25",  # pytype 2021.6.17 complains on recurrent.py, need to investigate!
        "@pip_pypi__sentencepiece//:pkg",  # WMT example.
        "@pip_pypi__svn//:pkg",
        # "tensorflow-cpu>=2.4.0",
        # "tensorflow_text>=2.4.0",  # WMT example.
        # "tensorflow_datasets",
        # "tensorflow==2.4.1",  # TODO(marcvanzee): Remove once #1326 is fixed.
    ],
)
