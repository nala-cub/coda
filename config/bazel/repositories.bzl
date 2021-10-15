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
""" Research repositories """

load("//tools:maybe_http.bzl", "http_archive")

def _clean_dep(x):
    return str(Label(x))

def _py_repositories():
    http_archive(
        name = "pytoolz_toolz",
        build_file = _clean_dep("//third_party:toolz.BUILD"),
        sha256 = "5c6ebde36ec2ceb9d6b3946105ba10b25237a67daee4eb80d62c508b9c4c2f55",
        strip_prefix = "toolz-0.11.1",
        urls = [
            "https://github.com/pytoolz/toolz/archive/0.11.1.tar.gz",
        ],
    )

    http_archive(
        name = "pytoolz_cytoolz",
        build_file = _clean_dep("//third_party:cytoolz.BUILD"),
        sha256 = "dba4a9d95e49f4f3cb5c41937f55dffe600aca5a7e640e3c2a56d9224923d7bb",
        strip_prefix = "cytoolz-0.11.0",
        urls = [
            "https://github.com/pytoolz/cytoolz/archive/0.11.0.tar.gz",
        ],
    )

    http_archive(
        name = "dm_tensor_annotations",
        build_file = _clean_dep("//third_party:tensor_annotations.BUILD"),
        patch_args = ["-p1"],
        patches = [Label("//third_party:tensor_annotations.patch")],
        sha256 = "d0a932efa70b1465860b14b5bbaf9b8eae8666133b28e74eaebdec9f30053f39",
        strip_prefix = "tensor_annotations-b24a6213d20e806d9f06f4af9e0c0d1707b26d3e",
        urls = [
            "https://github.com/deepmind/tensor_annotations/archive/b24a6213d20e806d9f06f4af9e0c0d1707b26d3e.tar.gz",
        ],
    )

    http_archive(
        name = "python_typeshed",
        build_file = _clean_dep("//third_party:typeshed.BUILD"),
        sha256 = "af75f84e1bbef6c3074307c144be09a79ba2ce52be71934e77eb9b5a05cf5796",
        strip_prefix = "typeshed-b2082ce5594f3f47630f6602f723afc768e3cc60",
        urls = [
            "https://github.com/python/typeshed/archive/b2082ce5594f3f47630f6602f723afc768e3cc60.tar.gz",
        ],
    )
    http_archive(
        name = "dm_rlax",
        build_file = _clean_dep("//third_party:rlax.BUILD"),
        sha256 = "d2283be962dc697882ff371813c64220a2c34a5538ca017d5bf699848426be3f",
        strip_prefix = "rlax-4e8aeed362d65ebb80bac162f09994c322c966a1",
        urls = ["https://github.com/deepmind/rlax/archive/4e8aeed362d65ebb80bac162f09994c322c966a1.tar.gz"],
    )

    http_archive(
        name = "dm_optax",
        build_file = _clean_dep("//third_party:optax.BUILD"),
        sha256 = "39a48c13be5e8259656dc7ed613dceaea9b205e1927b8b87db3c0e8181f18739",
        strip_prefix = "optax-0.0.9",
        urls = ["https://github.com/deepmind/optax/archive/v0.0.9.tar.gz"],
    )

    http_archive(
        name = "dm_chex",
        build_file = _clean_dep("//third_party:chex.BUILD"),
        sha256 = "c3d7029bc4225086822d3510b00f60cc28f6c22391e7fffdf1cea1634c3e49e2",
        strip_prefix = "chex-b00834cefcd8c50dc696a82f02ab92ed399bf530",
        urls = ["https://github.com/deepmind/chex/archive/b00834cefcd8c50dc696a82f02ab92ed399bf530.tar.gz"],
    )

    http_archive(
        name = "com_google_flax",
        build_file = _clean_dep("//third_party:flax.BUILD"),
        sha256 = "b0da699b317fe028f6b0ae94174ec0a17ca376a79ca0a48e5b106ee7070d849c",
        strip_prefix = "flax-0.3.5",
        urls = ["https://github.com/google/flax/archive/v0.3.5.tar.gz"],
    )
    http_archive(
        name = "dm_tree",
        build_file = _clean_dep("//third_party:tree.BUILD"),
        sha256 = "542449862e600e50663128a31cd4e262880f423f8bc66a64748f9bb20762cfbe",
        strip_prefix = "tree-42e87fda83278e2eb32bb55225e1d1511e77c10c",
        urls = ["https://github.com/deepmind/tree/archive/42e87fda83278e2eb32bb55225e1d1511e77c10c.tar.gz"],
    )
    http_archive(
        name = "dm_fancyflags",
        build_file = _clean_dep("//third_party:fancyflags.BUILD"),
        sha256 = "19805c12d7512c9e2806c0a6fea352381b4718e25d94d94960e8f3e61e3e4ab2",
        strip_prefix = "fancyflags-2e13d9818fb41dbb4476c4ebbcfe5f5a35643ef0",
        url = "https://github.com/deepmind/fancyflags/archive/2e13d9818fb41dbb4476c4ebbcfe5f5a35643ef0.tar.gz",
    )

    http_archive(
        name = "hf_transformers",
        build_file = _clean_dep("//third_party/py:transformers.BUILD"),
        patch_args = ["-p1"],
        patches = [_clean_dep("//third_party/py:transformers.patch")],
        sha256 = "30d9e30583e47680fd7b9809138c4cd83166fa0770f0113a1e06c3f65b848b4d",
        strip_prefix = "transformers-4.10.3",
        urls = [
            "https://github.com/huggingface/transformers/archive/v4.10.3.tar.gz",
        ],
    )

def _coda_repositories():
    http_archive(
        name = "com_github_openai_clip",
        build_file = _clean_dep("//third_party:clip.BUILD"),
        sha256 = "89cc8c65431d4f97abf99be30036131eaa9d1236fd684450c7eddb6f78003e15",
        strip_prefix = "CLIP-04f4dc2ca1ed0acc9893bd1a3b526a7e02c4bb10",
        urls = ["https://github.com/openai/CLIP/archive/04f4dc2ca1ed0acc9893bd1a3b526a7e02c4bb10.tar.gz"],
    )

    http_archive(
        name = "com_github_willwhitney_reprieve",
        build_file = _clean_dep("//third_party:reprieve.BUILD"),
        sha256 = "5d8e3ae90582a82f5e1f9dc65b007e9556048c2c728e85c8c4d80fa82258794a",
        strip_prefix = "reprieve-004e09a37e3c595c450ab05342cd779fa28be462",
        urls = ["https://github.com/willwhitney/reprieve/archive/004e09a37e3c595c450ab05342cd779fa28be462.tar.gz"],
    )

def research_repositories():
    """ Research repositories """

    # Override tensorflow @rules_python version. As of 2021-09-21, the only
    # target for which tensorflow uses @rules_python is:
    # @org_tensorflow//tensorflow/platform/python/platform:platform
    # This uses @rules_python//python/runfiles, which still exists in v0.4.0.
    http_archive(
        name = "rules_python",
        sha256 = "954aa89b491be4a083304a2cb838019c8b8c3720a7abb9c4cb81ac7a24230cea",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
            "https://github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
        ],
    )

    ############################################################################
    # JAX & Tensoflow
    http_archive(
        name = "org_tensorflow",
        patch_args = ["-p1"],
        patches = [
            "@com_google_jax//third_party:tensorflow.patch",
            Label("//third_party:tensorflow-sqlite.patch"),
            Label("//third_party:tensorflow-pyconfig.patch"),
        ],
        sha256 = "6b14b66a74728736359afcb491820fa3e713ea4a74bff0defe920f3453a3a0f0",
        strip_prefix = "tensorflow-b5b1ff47ad250c3e38dcadef5f6bc414b0a533ee",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/b5b1ff47ad250c3e38dcadef5f6bc414b0a533ee.tar.gz",
        ],
    )

    http_archive(
        name = "com_google_jax",
        sha256 = "a2f6e35e0d1b5d2bed88e815d27730338072601003fce93e6c49442afa3d8d96",
        strip_prefix = "jax-c3bacb49489aac6eb565611426022b3dd2a430fa",
        urls = [
            "https://github.com/corypaik/jax/archive/c3bacb49489aac6eb565611426022b3dd2a430fa.tar.gz",
        ],
    )

    ############################################################################
    http_archive(
        name = "bazel_gazelle",
        sha256 = "62ca106be173579c0a167deb23358fdfe71ffa1e4cfdddf5582af26520f1c66f",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.23.0/bazel-gazelle-v0.23.0.tar.gz",
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.23.0/bazel-gazelle-v0.23.0.tar.gz",
        ],
    )

    http_archive(
        name = "com_github_bazelbuild_buildtools",
        sha256 = "b8b69615e8d9ade79f3612311b8d0c4dfe01017420c90eed11db15e9e7c9ff3c",
        strip_prefix = "buildtools-4.2.1",
        url = "https://github.com/bazelbuild/buildtools/archive/4.2.1.tar.gz",
    )

    # we rely on dbx_build_tools for the inbuild python interpreter deps.
    http_archive(
        name = "dbx_build_tools",
        patch_args = ["-p1"],
        sha256 = "1076b50f33b093a6e5fc8bfccfd8ad32394631beb14caed4b62896b0953fa46f",
        strip_prefix = "dbx_build_tools-b735dc993d8e36bc95f9e9a3c9f1af2b45718825",
        urls = ["https://github.com/dropbox/dbx_build_tools/archive/b735dc993d8e36bc95f9e9a3c9f1af2b45718825.tar.gz"],
    )

    http_archive(
        name = "facebook_zstd",
        build_file_content = """exports_files(["zstd"])""",
        patch_cmds = ["make zstd"],
        sha256 = "5194fbfa781fcf45b98c5e849651aa7b3b0a008c6b72d4a0db760f3002291e94",
        strip_prefix = "zstd-1.5.0",
        urls = ["https://github.com/facebook/zstd/releases/download/v1.5.0/zstd-1.5.0.tar.gz"],
    )

    http_archive(
        name = "io_bazel_stardoc",
        sha256 = "cd3d1e483eddf9f73db2bd466f329e1d10d65492272820eda57540767c902fe2",
        strip_prefix = "stardoc-0.5.0",
        urls = ["https://github.com/bazelbuild/stardoc/archive/0.5.0.tar.gz"],
    )

    # Overwrite @dbx_build_tools version of cpython3.8. Note that we use the
    # same version, just with a different BUILD file. We could (and used to)
    # just use a patch, but it becomes frustrating to make fixes and we'd like
    # to avoid another having yet another submodule.
    http_archive(
        name = "org_python_cpython_38",
        build_file = _clean_dep("//third_party/cpython:python38.BUILD"),
        sha256 = "75894117f6db7051c1b34f37410168844bbb357c139a8a10a352e9bf8be594e8",
        strip_prefix = "Python-3.8.1",
        urls = ["https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tar.xz"],
    )

    _py_repositories()

    # for specific projects
    _coda_repositories()
