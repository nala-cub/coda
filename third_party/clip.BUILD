# CLIP BUILD
#   For: https://github.com/openai/CLIP
#

load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

exports_files([
    "LICENSE",
    "CLIP.png",
])

py_library(
    name = "clip",
    srcs = glob(["clip/*.py"]),
    data = ["clip/bpe_simple_vocab_16e6.txt.gz"],
    srcs_version = "PY3",
    visibility = [
        "@com_github_corypaik_coda//projects/coda:internal",
    ],
    deps = [
        "@pip_pypi__ftfy//:pkg",
        "@pip_pypi__regex//:pkg",
        "@pip_pypi__torch//:pkg",  # 1.7.1
        "@pip_pypi__torchvision//:pkg",  # 0.8.2
        "@pip_pypi__tqdm//:pkg",
    ],
)
