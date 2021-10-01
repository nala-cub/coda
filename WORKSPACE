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
workspace(name = "com_github_corypaik_coda")

# Fetch direct repositories
load("//config/bazel:repositories.bzl", "research_repositories")

research_repositories()

# Initialize TensorFlow's dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

# Fetch indirect dependecies and setup toolchain repos.
load("//config/bazel:workspace.bzl", "research_workspace")

research_workspace()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip",
    python_interpreter_target = "@pip_py3_interpreter//:python_bin",
    requirements_lock = "//third_party:requirements.txt",
)

load("@pip//:requirements.bzl", "install_deps")

install_deps()
