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
""" Configures the in-build python interpreter.

This is a non-detecting in-build version of @pybind11's //:python_configure.bzl
compatible with `tensorflow` and `jax`.

Limitations
* `:python_embed` has not been used or validated, so may not work. the pybind11
  repo has extra flags associated with this target, but it's unclear to me which
  are necessary here, or how we might test that. Once we find a library which
  utilizes it that may help to debug any issues that arise.
* :`numpy_headers` is a bit hacky, and relies on installing the numpy package
  via `@rules_python` and copying over the header files.

Add the following to your WORKSPACE file:
```python
load("@com_github_corypaik_coda//tools/python:configure.bzl", "python_configure")
python_configure(name = "research_config_python")
```

One can automatically use this on a compatible system by adding the following
flags to their `.bazelrc`:
```
common:linux --repo_env=TF_PYTHON_CONFIG_REPO="@research_config_python"
common:linux --repo_env=TF_LOCAL_PYTHON_CONFIG_REPO="@research_config_python"
```
These two flags will override the `@local_config_python` and
`@local_execution_config_python` repositories.
"""

def _python_configure_impl(repository_ctx):
    """Implementation of the python_configure repository rule."""
    repository_ctx.template("BUILD", Label("//tools/python:configure.BUILD"), {})

python_configure = repository_rule(
    implementation = _python_configure_impl,
)
