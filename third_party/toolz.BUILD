# Description:
#   A set of utility functions for iterators, functions, and dictionaries.
#   See the PyToolz documentation at https://toolz.readthedocs.io

load(
    "@com_github_corypaik_coda//tools:defaults.bzl",
    "py_generate_test_suite",
    "py_library",
)

licenses(["notice"])  # New BSD

exports_files(["LICENSE"])

py_library(
    name = "tlz",
    srcs = glob(["tlz/*.py"]),
    imports = ["."],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

# NOTE: We include tlz as a dependency to mirror the pip package.
py_library(
    name = "toolz",
    srcs = glob(
        ["toolz/**/*.py"],
        exclude = [
            "toolz/tests/**/*.py",
        ],
    ),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":tlz"],
)

py_generate_test_suite(
    name = "tests",
    srcs = glob(
        ["toolz/tests/**/*"],
    ),
    deps = [
        ":toolz",
    ],
)
