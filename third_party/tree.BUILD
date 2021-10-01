# Description:
# Tree provides utilities for working with nested data structures. This build
# file was adopted from the last commit prior to their switch to cmake.
# Source:
#  https://github.com/deepmind/tree/blob/2305ebc528f9e8e62641a20e59912971ec650a5d/tree/BUILD

load("@com_github_corypaik_coda//tools:defaults.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])  # Apache-2.0

exports_files(["LICENSE"])

py_library(
    name = "tree",
    srcs = ["tree/__init__.py"],
    data = [
        "tree/_tree.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":sequence",
    ],
)

py_library(
    name = "sequence",
    srcs = [
        "tree/sequence.py",
    ],
    data = [
        "tree/_tree.so",
    ],
    srcs_version = "PY2AND3",
)

pybind_extension(
    name = "tree/_tree",
    srcs = [
        "tree/tree.cc",
        "tree/tree.h",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    module_name = "_tree",
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

py_test(
    name = "tree_test",
    srcs = ["tree/tree_test.py"],
    deps = [
        ":tree",
        "@absl_py//absl/testing:parameterized",
        "@pip_pypi__attrs//:pkg",
        "@pip_pypi__numpy//:pkg",
        "@wrapt",
    ],
)

py_test(
    name = "tree_benchmark",
    srcs = ["tree/tree_benchmark.py"],
    deps = [":tree"],
)
