# tensor_annotations BUILD
#   For: https://github.com/deepmind/tensor_annotations

load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])  # Apache-2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "tensor_annotations",
    srcs = glob(
        ["tensor_annotations/**/*.py"],
        exclude = [
            "tensor_annotations/tests/**/*",
        ],
    ),
    data = ["@python_typeshed//:tensor_annotation_files"],
    srcs_version = "PY3",
)

exports_files(glob(["**/*.pyi"]))

# Can't seem to get the tests to work without a ton of import errors. Seems to
# be a common problem, Haiku had the errors due to the `typing` module. In this
# case it's that the test is `jax.py`, which collides with the package `jax`.
# Probably worth revisiting at some point. The tests run fine outside of Bazel,
# so maybe something to do with the PYTHONPATH.

# py_pytest_test(
#     name = "tensor_annotations_test",
#     srcs = glob(["tensor_annotations/tests/*.py"]),
#     pytest_location = "external/dm_tensor_annotations",
#     deps = [":tensor_annotations"],
# )

# py_test(
#     name = "tensor_annotations_jax_test",
#     main = "tensor_annotations/tests/jax.py",
#     srcs = ["tensor_annotations/tests/jax.py"],
#     deps = [
#         ":tensor_annotations",
#         "@pip_pypi__absl_py//:pkg",
#         "@pip_pypi__jax//:pkg",
#     ],
# )
