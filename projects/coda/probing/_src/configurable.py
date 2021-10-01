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
""" Provides a configurable decorator for fancyflags. """
from functools import wraps
import inspect
from typing import Callable, Optional, TypeVar

from absl import flags
import fancyflags as ff

_T = TypeVar('_T')

flags.disclaim_key_flags()


def _make_configurable(
    fn_or_cls: Callable[..., _T],
    name: Optional[str],
    module: Optional[str],
) -> Callable[..., _T]:

  # name is none, use function name
  if name is None:
    name = fn_or_cls.__name__

  # if module is provided, will be model.name
  if module is not None:
    name = '.'.join([module, name])

  doc_str = fn_or_cls.__doc__ or ''
  doc_str_lines = doc_str.splitlines()
  doc_str_header = doc_str_lines[0] if len(doc_str_lines) > 0 else ''
  ff_fn_partial_def = ff.DEFINE_auto(name, fn_or_cls, doc_str_header)

  # ArgSpec(args, varargs, keywords, defaults)
  # note: DEFINE_auto doesn't support varargs or keywords. Additionally,
  # all args MUST have a default.
  spec = inspect.getfullargspec(fn_or_cls)
  arg_keys = spec.args

  @wraps(fn_or_cls)
  def _configurable(*args, **kwargs):
    ff_fn_partial = ff_fn_partial_def.value
    # the partial will have all defaults defined as kwargs, so we need to
    # convert any args to kwargs to maintain functionality.
    num_args = len(args)
    if num_args > len(arg_keys):
      raise ValueError(
          f'Cannot call function {name} with {num_args} arguments, see '
          f'signature: \n {fn_or_cls.__code__}')

    # split the valid keys
    arg_arg_keys, kwarg_arg_keys = arg_keys[:num_args], arg_keys[num_args:]

    args_keyed = dict(zip(arg_arg_keys, args))
    assert set(kwargs) | set(kwarg_arg_keys) == set(kwarg_arg_keys), (
        'Found overlap between args and kwarg keys.',
        set(kwargs) | set(kwarg_arg_keys),
        set(kwarg_arg_keys),
    )
    return ff_fn_partial(**{**args_keyed, **kwargs})

  return _configurable


def configurable(
    name_or_fn=None,
    module: Optional[str] = None,
):
  """ Decorator to make a function or class configurable with fancyflags.

  While much more limited in functionality this is partially inspired by
    https://github.com/google/gin-config/blob/master/gin/config.py

  This decorator is meant to  decorator is meant to be backward compatible with
  gin, though it leverages fancyflags instead of gin on the backend. This means
  that you no longer have to use the gin binding syntax with absl flags of
  `--gin_bind="my_configurable.my_param='the'"`, which tends to have very bad
  integration with other argument passing libraries (think tuning libs etc.).

  Instead the configurable is much simpler, and equivalent to defining your fn
  and writing:
    >>> MY_CONFIGURED_FN = ff.DEFINE_auto('name', my_fn, 'help_msg')
  This is limited to all the fancyflags types (not gin), but it'd be interesting
  to merge these two libraries deeper in the future to get more of gin's
  compatibility with fancyflags ease of cmd use.

  Returns:
    When used with no parameters (or with a function/class supplied as the first
    parameter), it returns the decorated function or class. When used with
    parameters, it returns a function that can be applied to decorate the target
    function or class.

  """

  # we may be calling with `@configurable` or `@configurable(<name>)`.
  decoration_target = None
  if callable(name_or_fn):
    decoration_target = name_or_fn
    name = None
  else:
    name = name_or_fn

  def perform_decoration(fn_or_cls: Callable[..., _T]) -> Callable[..., _T]:
    return _make_configurable(fn_or_cls, name, module)

  if decoration_target:
    return perform_decoration(decoration_target)
  return perform_decoration
