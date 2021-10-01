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
""" A low dependence collection of general utilities.

  This file relies on third-party packages for specific functionalities. Some
  of these are givens, e.g. you won't need to convert Torch Tensors -> lists
  unless PyTorch is installed. Others, such as loading yml files, do require
  specific packages. See the error messages or `@require('<pkg>')` decorators
  for details on what packages are required.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from collections.abc import Sequence
from contextlib import contextmanager
from functools import lru_cache
from functools import wraps
import gc
import hashlib
import importlib
from importlib.util import find_spec
import inspect
import itertools
import json
from pathlib import Path
import re
import time
from types import ModuleType
from typing import Any, Callable, Tuple, Type, TypeVar, Union, overload
import warnings

try:
  from typing import Literal
except ImportError:  # python 3.7
  from typing_extensions import Literal
# no typing_extensions
except ImportError:  # pylint: disable=duplicate-except
  Literal = list

from absl import logging
import cytoolz.curried as T

El = TypeVar('El')
ExceptException = Union[Type[BaseException], Tuple[Type[BaseException]]]


@lru_cache(maxsize=None)
def maybe_import(name: str) -> Union[ModuleType, None]:
  """ Imports a package if installed

  Args:
    name: Name of the package

  Returns:
    package `name`, if installled
    otherwise returns `None`

  """
  return importlib.import_module(name) if is_installed(name) else None


@lru_cache(maxsize=None)
def is_installed(name: str) -> bool:
  """ Checks if a module is installed.
  Args:
    name: Name of the module. If name is for a submodule (contains a dot) and
      the parent module is not installed, `importlib.util.find_spec` will raise
      a `ModuleNotFoundError`. We catch these errors and return False.
  """
  try:
    return find_spec(name) is not None
  except ModuleNotFoundError:
    return False
  except ValueError as inst:
    if str(inst).endswith('.__spec__ is None'):
      logging.info('Missing __spec__ for %s. Marking package as installed.',
                   str(inst).split('.', 1)[0])
      return True
    raise inst
  except:  # pylint: disable=bare-except
    logging.exception('Unhandled exception for is_installed, returning False')
    return False


def require(*names: str, as_arg=False):
  """Create a decorator to check if a package is installed.

  Args:
    *names: Name of the package(s).

  """
  msg = '%s requires %s, but %s not installed.'

  def wrapped_with_params(fn):

    @wraps(fn)
    def _require(*args, **kwargs):
      missing_pkgs = list(T.filter(T.complement(is_installed), names))
      if missing_pkgs == []:
        if as_arg:
          return fn(*T.map(maybe_import, names), *args, **kwargs)
        # if is_installed(name):
        return fn(*args, **kwargs)
      elif len(missing_pkgs) == 1:
        raise ImportError(msg % (fn.__name__, missing_pkgs[0], 'it is'))
      else:
        nice_names = ', '.join(missing_pkgs)
        raise ImportError(msg % (fn.__name__, nice_names, 'they are'))

    return _require

  return wrapped_with_params


@contextmanager
def catch_exp_failures(
    name: str,
    verbose: bool = True,
    exit_on: ExceptException = KeyboardInterrupt,
    catch: ExceptException = Exception,
):
  """ Context manager for catching experiment failures.

  This context manager will ignore all exceptions within context except
  Keyboard Interrupts and is useful for running sets of experiments. Failed
  experiments will be logged to stderr (absl) and skipped. If torch is
  installed, we empty the catch on exit of the context manager. We also run
  Garbage collection. This should cleanup most GPU allocations, and is known
  to work with Jax, PyTorch, and DALI.

  Examples:
    >>> with catch_exp_failures('Model'):
    ...   lambda : raise RuntimeError
    Failed to run Model after * s. Skipping...
    >>> with catch_exp_failures('Model'):
    ...   lambda : None
    Successfully ran Model in * s.

  Args:
    name: Name of the experiment
    verbose: Predicate indicating whether to log details about timing
      information of the experiment on success.
    exit_on: Exception type(s) to exit on, defaults to Keyboard Interruptions
      only.
    catch: Exception type(s) to catch, defaults to all exceptions.
  """
  tick = time.time()
  try:
    yield
    if verbose:
      logging.info('Successfully ran %s in %0.3d s.', name, time.time() - tick)
  except catch as e:  # pylint: disable=broad-except
    # exit_on may be a subset of catch, in which case we should exit here.
    if isinstance(e, exit_on):
      raise e
    logging.exception('Failed to run %s after %s s. Skipping...', name,
                      time.time() - tick)
  finally:
    # maybe empty torch cuda cache
    torch = maybe_import('torch')
    if torch is not None:
      torch.cuda.empty_cache()
    gc.collect()


def tolist(x) -> list:
  """ Convert a list-like object to a python list.

  Converts a list-like object into a python list. This function requires no
  additional packages to be installed, and checks each input using a string of
  the type. This can be useful for reliably converting tensors to a format
  supported by json or other logging libraries.

  Args:
    x: A list-like object, which can be a torch Tensor, numpy array, or jax
      array. If `x` is not any of these, the default is to try `x.tolist()`, or
      just return `x` if that fails.
  """
  x_type = str(type(x))
  if x_type == '<class \'torch.Tensor\'>':
    x = x.tolist()
  elif x_type.endswith('DeviceArray\'>'):
    onp = maybe_import('numpy')
    if onp is not None:
      x = onp.asarray(x).tolist()
  else:
    try:
      x = x.tolist()
    except (AttributeError, TypeError):
      pass
  return x


def topylist(x) -> list:
  warnings.warn('labtools.topylist is deprecated, use labtools.tolist instead',
                DeprecationWarning)
  return tolist(x)


def compute_obj_hash(obj) -> str:
  """ Computes the hash of an object.

  This uses `labtools.BestEffortJsonEncoder` to dump `obj` as a json string and
  uses SHA256 to hash that string. This function is meant to hash any input by
  representing it as a string.

  """
  str_obj = json.dumps(obj, cls=BestEffortJSONEncoder, sort_keys=True)
  return hashlib.sha256(str_obj.encode('utf-8')).hexdigest()


class CustomJSONEncoder(json.JSONEncoder):
  """JSON encoder w/ support for ConfigDicts, Paths, and Arrays.
  Note:
    This is based off of ml_collections.CustomJSONEncoder, with added support
    for Paths and Arrays (it also doesn't require ml_collections)
  """

  def default(self, o):
    # maybe handle config dicts

    ml_collections = maybe_import('ml_collections')
    if ml_collections is not None:
      if isinstance(o, ml_collections.FieldReference):
        return o.get()
      elif isinstance(o, ml_collections.ConfigDict):
        return o._fields
    # paths
    if isinstance(o, Path):
      return str(o)
    # lists
    else:
      o = topylist(o)
      if isinstance(o, list):
        return o
      raise TypeError(
          '{} is not JSON serializable. Instead use cls=BestEffortJSONEncoder'.
          format(type(o)))


class BestEffortJSONEncoder(CustomJSONEncoder):
  """Best Effort JSON encoder (won't throw errors).
  Source:
    ml_collections._BestEffortCustomJSONEncoder
  """

  def default(self, o):
    try:
      return super().default(o)
    except TypeError:
      if isinstance(o, set):
        return sorted(list(o))
      elif inspect.isfunction(o):
        return 'function {}'.format(o.__name__)
      elif (hasattr(o, '__dict__') and o.__dict__ and
            not inspect.isclass(o)):  # Instantiated object's variables
        return dict(o.__dict__)
      elif hasattr(o, '__str__'):
        return 'unserializable object: {}'.format(o)
      else:
        return 'unserializable object of type: {}'.format(type(o))


@overload
def ensure_listlike(x: Sequence[El]) -> Sequence[El]:
  ...


@overload
def ensure_listlike(x: str) -> list[str]:
  ...


@overload
def ensure_listlike(x: El) -> list[El]:
  ...


def ensure_listlike(x):
  return x if isinstance(x, Sequence) and not isinstance(x, str) else [x]


def _check_arg_lens(args):
  args = list(map(list, args))
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))


def safe_zip(*iterables):
  """ Checks for matching lengths before zipping multiple iterables.

  Note:
    This function is not streaming, if any of `iterables` are generaters, they
    will be fully evaluated first.

  Args:
    *iterables: Iterables to check and aggregate
  """
  _check_arg_lens(iterables)
  return list(zip(*iterables))


def safe_map(f: Callable, *iterables):
  """ Checks for matching lengths before mapping iterables.

  Note:
    This function is not streaming, if any of `iterables` are generaters, they
    will be fully evaluated first.

  Args:
    f: function to apply to each element of the iterables.
    *iterables: iterables to be mapped

  """
  _check_arg_lens(iterables)
  return list(map(f, *iterables))


def unzip(obj):
  return list(zip(*obj))


@overload
def flatten_dict(  # pytype: disable=invalid-annotation
    d: dict[str, Any],
    parent_key: str = '',
    sep: str = '/',
    sort: Literal[True] = True) -> OrderedDict:
  ...


@overload
def flatten_dict(  # pytype: disable=invalid-annotation
    d: dict[str, Any],
    parent_key: str = '',
    sep: str = '/',
    sort: Literal[False] = True) -> dict:
  ...


def flatten_dict(d, parent_key='', sep='/', sort=True):
  """ Flatten a nested dictionary

  Flattens a nested dictionary by joining keys at each depth using `sep`.

  Examples:
    >>> flatten_dict({'a': {'b': 2}, c: 1})
        {'a.b': 2, 'c':1}
  """
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, MutableMapping):
      items.extend(flatten_dict(v, new_key, sep=sep, sort=False).items())
    else:
      items.append((new_key, v))
  if sort:
    return OrderedDict(sorted(items, key=lambda x: x[0]))
  return dict(items)  # pytype: disable=bad-return-type


def split_by_keys(obj: dict[str, El],
                  keys: list[str]) -> Tuple[dict[str, El], dict[str, El]]:
  # split dict into 2, where first only contains keys in `keys`.
  return {k: obj.pop(k) for k in keys}, obj


@require('numpy')
def get_differences(x, y) -> str:
  """ Get differences of arrays (for debugging)

  Adapted from `numpy.testing._private.utils.assert_array_compare`[1]_

  Args:
    x (np.ndarray): Array 1 to compare
    y (np.ndarray): Array 2 to compare

  References:
  .. [1] github.com/numpy/numpy/blob/main/numpy/testing/_private/utils.py
  """
  np = maybe_import('numpy')

  remarks = []
  error = np.abs(x - y)
  max_abs_error = np.max(error)
  remarks.append('Max absolute difference: %f' % max_abs_error)

  # note: this definition of relative error matches that one
  # used by assert_allclose (found in np.isclose)
  # Filter values where the divisor would be zero
  nonzero = np.bool_(y != 0)
  if np.all(~nonzero):
    max_rel_error = np.array(np.inf)
  else:
    max_rel_error = np.max(error[nonzero] / np.abs(y[nonzero]))
  remarks.append('Max relative difference: %f' % max_rel_error)
  return '\n'.join(remarks)


@require('absl.testing.parameterized')
def param_product(**testgrid):
  """A decorator for running tests over cartesian product of parameters values.

  Backport from `absl.testing.parameterized.product` for use in `absl` <
    v0.12.0 [1]_.

  See the module docstring [1]_ for a usage example. The test will be run for
  every possible combination of the parameters.

  Args:
    **testgrid: A mapping of parameter names and their possible values.
      Possible values should given as either a list or a tuple.
  Raises:
    NoTestsError: Raised when the decorator generates no tests.
  Returns:
      A test generator to be handled by TestGeneratorMetaclass.
  References:
  .. [1] github.com/abseil/abseil-py/blob/master/absl/testing/parameterized.py
  """
  parameterized = maybe_import('absl.testing.parameterized')

  for name, values in testgrid.items():
    if not isinstance(values, (list, tuple)):
      raise ValueError(
          'Values of {} must be given as list or tuple, found {}'.format(
              name, type(values)))

  # Create all possible combinations of parameters as a cartesian product
  # of parameter values.
  testcases = [
      dict(zip(testgrid.keys(), product))
      for product in itertools.product(*testgrid.values())
  ]

  return parameterized._parameter_decorator(  # pylint: disable=protected-access
      parameterized._ARGUMENT_REPR,  # pylint: disable=protected-access
      testcases,
  )


def cleanup_cache_files(cachedir: Path, pattern: str, dry_run: bool = False):
  for spath in cachedir.iterdir():
    if re.match(pattern, spath.name) is not None:
      logging.info('Removing cached file at %s (pattern=%s)', spath, pattern)
      if not dry_run:
        spath.unlink()
    else:
      logging.debug('Skipping cached file at %s (pattern=%s)', spath, pattern)
