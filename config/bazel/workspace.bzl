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
""" Research workspace """

load("@com_google_jax//third_party/pocketfft:workspace.bzl", pocketfft = "repo")
load("@dbx_build_tools//build_tools/bazel:external_workspace.bzl", "drte_deps")
load("@io_bazel_stardoc//:setup.bzl", "stardoc_repositories")
load("//tools/python:configure.bzl", "python_configure")
load("//tools/python:interpreter.bzl", "py3_interpreter")

def research_workspace():
    """ Research workspace """

    # we rely on dbx_build_tools for the in-build python interpreter deps.
    drte_deps()

    # build @pip_py3_interpreter for running pip_install or pip_parse.
    # This interpreter is not fully functional, and installed as a patch.
    py3_interpreter(name = "pip_py3_interpreter")

    # Configure the in-build python config repo.
    # This is a much more complex python toolchain adopted from @dbx_build_tools.
    # It's built from with custom build rules and works very well. The caveat is
    # the binary is not available for workspace rules, so it can't be used as
    # the interpreter for pip_install.
    python_configure(name = "research_config_python")

    # jax
    pocketfft()

    stardoc_repositories()
