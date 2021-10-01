load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])  # Apache-2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "fancyflags",
    srcs = glob(
        ["fancyflags/**/*.py"],
        exclude = ["**/*_test.py"],
    ),
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl/flags",
    ],
)
