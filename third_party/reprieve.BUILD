# Reprieve BUILD
#   For: https://github.com/willwhitney/reprieve

load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

exports_files(["README.md"])

# version comments are estimates
py_library(
    name = "reprieve",
    srcs = glob(["reprieve/**/*.py"]),
    srcs_version = "PY3",
    visibility = [
        "@com_github_corypaik_coda//projects/coda:internal",
    ],
    deps = [
        "@pip_pypi__altair//:pkg",
        "@pip_pypi__altair_saver//:pkg",
        "@pip_pypi__pandas//:pkg",
        "@pip_pypi__selenium//:pkg",
        "@pip_pypi__torch//:pkg",  # 1.7.1 << seems to work
        "@pip_pypi__torchvision//:pkg",  # 0.8.2 << seems to work
    ],
)
