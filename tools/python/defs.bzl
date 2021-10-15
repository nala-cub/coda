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
""" PyType Integration Macros"""

load(
    "@rules_python//python:defs.bzl",
    _py_binary = "py_binary",
    _py_library = "py_library",
    _py_test = "py_test",
)

def py_binary(name, python_version = "PY3", srcs_version = "PY3", **kwargs):
    _py_binary(
        name = name,
        srcs_version = srcs_version,
        python_version = python_version,
        **kwargs
    )

def py_library(name, srcs_version = "PY3", **kwargs):
    _py_library(
        name = name,
        srcs_version = srcs_version,
        **kwargs
    )

def py_test(name, python_version = "PY3", srcs_version = "PY3", **kwargs):
    _py_test(
        name = name,
        srcs_version = srcs_version,
        python_version = python_version,
        **kwargs
    )

def py_generate_test_suite(
        name,
        srcs,
        sizes = {},
        default_size = "medium",
        **kwargs):
    """ Creates a test suite containing a py_test for each of srcs

    Args:
        name: Name for the test suite
        srcs: list of source files, one for each py_test.
        sizes: Mapping of size overrides for individual tests.
        default_size: default test size
        **kwargs: Keyword arguments passed to py_test
    """
    generate_tests = {
        file.split(".py")[0]: file
        for file in srcs
    }

    for fname, file in generate_tests.items():
        py_test(
            name = fname,
            size = sizes.get(fname, default_size),
            srcs = [file],
            **kwargs
        )

    native.test_suite(
        name = name,
        tests = generate_tests.keys(),
    )

def _clean_dep(x):
    return str(Label(x))

def pytype_binary(name, pytype_deps = [], strict = False, **kwargs):
    """Proxy for py_binary that implicitly creates a PyType test.

    Args:
       name: Name for the py_binary rule.
       pytype_deps: A list of pytype-only deps
       strict: Predicate indicating that the pytype test should be strict.
       **kwargs: Keyword arguments passed to pytype_genrunle
    """

    py_binary(name = name, **kwargs)
    _pytype_genrule(
        name = name,
        pytype_deps = pytype_deps,
        strict = strict,
        **kwargs
    )

def pytype_strict_binary(**kwargs):
    pytype_binary(strict = True, **kwargs)

def pytype_library(name, pytype_deps = [], strict = False, **kwargs):
    """Proxy for py_library that implicitly creates a PyType test.

    Args:
       name: name for the py_binary rule.
       pytype_deps: a list of pytype-only deps
       strict: Predicate indicating that the pytype test should be strict.
       **kwargs: Keyword arguments passed to pytype_genrunle
    """

    py_library(name = name, **kwargs)
    _pytype_genrule(
        name = name,
        pytype_deps = pytype_deps,
        strict = strict,
        **kwargs
    )

def pytype_strict_library(**kwargs):
    pytype_library(strict = True, **kwargs)

def _pytype_genrule(
        name,
        pytype_deps,
        pytype_args = [
            "-x=external/",
            " .",
        ],
        strict = False,
        **kwargs):
    """A macro that runs pytest tests by using a test runner.

    Args:
        name: A unique name for this rule.
        pytype_deps: A list of pytype-only deps
        pytype_args: A list of arguments passed to pytype.
        strict: Predicate indicating that the pytype test should be strict.
        **kwargs: are passed to py_test, with srcs and deps attrs modified
    """
    kwargs.pop("main", [])
    deps = kwargs.pop("deps", []) + [_clean_dep("//tools/python:pytype_helper")]
    srcs = kwargs.pop("srcs", []) + [_clean_dep("//tools/python:pytype_helper")]
    args = kwargs.pop("args", [])
    args = pytype_args

    # maybe strict
    if strict:
        args = args + ["--strict-import"]

    # add pytype tag
    tags = kwargs.pop("tags", []) + ["pytype"]

    # add pytype deps
    deps += pytype_deps

    py_test(
        name = "%s.pytype" % name,
        srcs = srcs,
        main = "pytype_helper.py",
        python_version = "PY3",
        deps = deps,
        args = args,
        tags = tags,
        **kwargs
    )

def py_repl(name, deps, **kwargs):
    """ Macro to create a python REPL

    Args:
        name: Name of the rule
        deps: Dependencies (python libraries) to make avalible in the python
            REPL.
        **kwargs: Keyword arguments passed to `py_binary`.

    Source:
        https://github.com/thundergolfer/example-bazel-monorepo
    """
    py_binary(
        name = name,
        srcs = [_clean_dep("//tools/python:py_repl")],
        main = "py_repl.py",
        deps = deps,
        **kwargs
    )

def jupyterlab_server(name = "jupyterlab", **kwargs):
    """ A macro for creating a Jupyterlab Server.

    Args:
      name: A unique name for this rule.
      **kwargs: are passed to py_binary
    """

    py_binary(
        name = name,
        srcs = [_clean_dep("//tools/python:jupyterlab_helper")],
        main = "jupyterlab_helper.py",
        **kwargs
    )
