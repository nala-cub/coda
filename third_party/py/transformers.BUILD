# Description:
#   Transformers provides thousands of pretrained models to perform tasks on
#   texts such as classification, information extraction, question answering,
#   summarization, translation, text generation and more in over 100 languages.
#   Its aim is to make cutting-edge NLP easier to use for everyone.

load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

alias(
    name = "hf_transformers",
    actual = "transformers",
)

py_library(
    name = "transformers",
    srcs = glob(
        ["src/transformers/**/*.py"],
    ),
    imports = ["src"],
    deps = [
        "@pip_pypi__filelock//:pkg",
        "@pip_pypi__huggingface_hub//:pkg",
        "@pip_pypi__numpy//:pkg",
        "@pip_pypi__packaging//:pkg",
        "@pip_pypi__pyyaml//:pkg",
        "@pip_pypi__regex//:pkg",
        "@pip_pypi__requests//:pkg",
        "@pip_pypi__sacremoses//:pkg",
        "@pip_pypi__tokenizers//:pkg",
        "@pip_pypi__tqdm//:pkg",
    ],
)
