# Copyright 2021 Cory Paik. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Configure a python binary for use with pip install / pip parse

Adapted with modifications from:
thundergolfer/example-bazel-monorepo/tools/build/bazel/py_toolchain/py_interpreter.bzl
"""

OSX_OS_NAME = "mac os x"
LINUX_OS_NAME = "linux"

def _py3_interpreter_impl(rctx):
    os_name = rctx.os.name.lower()

    # TODO(Jonathon): This can't differentiate ARM (Mac M1) from old x86.
    # TODO(Jonathon: Support Windows.
    if os_name == OSX_OS_NAME:
        url = "https://github.com/indygreg/python-build-standalone/releases/download/20210228/cpython-3.8.8-x86_64-apple-darwin-pgo+lto-20210228T1503.tar.zst"
        integrity_shasum = "4c859311dfd677e4a67a2c590ff39040e76b97b8be43ef236e3c924bff4c67d2"
    elif os_name == "linux":
        url = "https://github.com/indygreg/python-build-standalone/releases/download/20210228/cpython-3.8.8-x86_64-unknown-linux-gnu-pgo+lto-20210228T1503.tar.zst"
        integrity_shasum = "74c9067b363758e501434a02af87047de46085148e673547214526da6e2b2155"
    else:
        fail("OS '{}' is not supported.".format(os_name))

    # TODO(corypaik): Just use download_and_extract when it supports zstd.
    # https://github.com/bazelbuild/bazel/pull/11968
    rctx.download(
        url = [url],
        sha256 = integrity_shasum,
        output = "python.tar.zst",
    )

    # Currently we fetch the repo in repositories.bzl and build the zstd binary
    # from a patch command. According to the readme this should work as long as
    # you have make installed, which is more common than zstd.
    zstd_bin_path = rctx.path(Label("@facebook_zstd//:zstd"))

    res = rctx.execute([zstd_bin_path, "-d", "python.tar.zst"])

    if res.return_code:
        fail("Error decompressiong with zstd" + res.stdout + res.stderr)

    rctx.extract(archive = "python.tar")
    rctx.delete("python.tar")
    rctx.delete("python.tar.zst")

    # NOTE: 'json' library is only available in Bazel 4.*.
    python_build_data = json.decode(rctx.read("python/PYTHON.json"))

    # Create a symlink python_bin -> python/.../python3.8
    rctx.symlink(
        "python/%s" % python_build_data["python_exe"],
        "python_bin",
    )

    BUILD_FILE_CONTENT = """
filegroup(
    name = "files",
    srcs = glob(["install/**"], exclude = ["**/* *"]),
    visibility = ["//visibility:public"],
)
filegroup(
    name = "interpreter",
    srcs = ["python/{interpreter_path}"],
    visibility = ["//visibility:public"],
)
""".format(interpreter_path = python_build_data["python_exe"])
    rctx.file("BUILD.bazel", BUILD_FILE_CONTENT)
    return None

py3_interpreter = repository_rule(
    implementation = _py3_interpreter_impl,
    attrs = {},
)
