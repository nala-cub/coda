# Description:
#   Cython implementation of the toolz package, which provides high performance
#   utility functions for iterables, functions, and dictionaries.

load(
    "@com_github_corypaik_coda//tools:defaults.bzl",
    "py_generate_test_suite",
    "pyx_library",
)

licenses(["notice"])  # New BSD

exports_files(["LICENSE"])

pyx_library(
    name = "cytoolz",
    srcs = glob(
        [
            "cytoolz/**/*.py",
            "cytoolz/**/*.pyx",
            "cytoolz/**/*.pxd",
        ],
        exclude = [
            "cytoolz/__init__.pxd",
        ],
    ),
    py_deps = [
        "@pytoolz_toolz//:toolz",
    ],
    visibility = ["//visibility:public"],
)

py_generate_test_suite(
    name = "tests",
    srcs = glob(
        ["cytoolz/tests/**/*.py"],
        exclude = ["cytoolz/tests/test_doctests.py"],
    ),
    deps = [
        ":cytoolz",
    ],
)
