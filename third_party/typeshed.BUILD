# typeshed BUILD
#   For: https://github.com/python/typeshed

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

licenses(["notice"])  # Apache-2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

filegroup(
    name = "files",
    srcs = glob(["**/*"]),
)

# HACK: Symlinks for `tensor_annotations`. AFAIK `copy_file` has to be called
# from the same directory. These are not included by default, so they will only
# be included when the library includes :tensor_annotation_files in the data
# field.
_TENSOR_ANNOTATIONS_TYPESHED_SYMLINKS = {
    "@dm_tensor_annotations//:jax-stubs/__init__.pyi": "stubs/jax/jax/__init__.pyi",
    "@dm_tensor_annotations//:jax-stubs/numpy/__init__.pyi": "stubs/jax/jax/numpy/__init__.pyi",
    "@dm_tensor_annotations//:tensor_annotations/__init__.py": "stubs/tensor_annotations/tensor_annotations/__init__.pyi",
    "@dm_tensor_annotations//:tensor_annotations/axes.py": "stubs/tensor_annotations/tensor_annotations/axes.pyi",
    "@dm_tensor_annotations//:tensor_annotations/jax.py": "stubs/tensor_annotations/tensor_annotations/jax.pyi",
    "@dm_tensor_annotations//:tensor_annotations/tensorflow.py": "stubs/tensor_annotations/tensor_annotations/tensorflow.pyi",
}

filegroup(
    name = "tensor_annotation_files",
    srcs = _TENSOR_ANNOTATIONS_TYPESHED_SYMLINKS.values(),
)

[
    copy_file(
        name = dest.replace("/", "."),
        src = src,
        out = dest,
        allow_symlink = True,
    )
    for src, dest in _TENSOR_ANNOTATIONS_TYPESHED_SYMLINKS.items()
]
