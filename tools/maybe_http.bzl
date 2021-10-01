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
""" Provides macros to skip existing repository rules

Simple wrappers around @bazel_tools//tools/build_defs/repo:http.bzl that
skip existing packages to retain compatibility with renovate. These rules
will provide warnings if a repo rule is skipped, a feature adapted from
tensorflow/tensorflow/third_party/repo.bzl.

"""

load(
    "@bazel_tools//tools/build_defs/repo:http.bzl",
    _http_archive = "http_archive",
    _http_file = "http_file",
)

def maybe(repo_rule, name, **kwargs):
    """ provides the functionality of built in maybe macro with warnings

    The warning is adapted from tensorflow/tensorflow/third_party/repo.bzl

    Args:
        repo_rule: repository rule function.
        name: name of the repository to create.
        **kwargs: remaining arguments that are passed to the repo_rule
            function.
    """

    # skip existing, but provide a warning message.
    if native.existing_rule(name):
        print(
            "\n\033[1;33mWARNING:\033[0m skipping import of repository '" +
            name + "' because it already exists.\n",
        )  # buildifier: disable=print
        return

    repo_rule(
        name = name,
        **kwargs
    )

def http_archive(name, **kwargs):
    """ http_archive wrapper that uses provides a warning message

    Args:
        name: name of the http_archive.
        **kwargs: remaining arguments that are passed to the `http_archive`
            function
    """

    maybe(
        _http_archive,
        name = name,
        **kwargs
    )

def http_file(name, **kwargs):
    """ http_file wrapper that uses provides a warning message

    Args:
        name: name of the http_file.
        **kwargs: remaining arguments that are passed to the `http_file`
            function
    """

    maybe(
        _http_file,
        name = name,
        **kwargs
    )
