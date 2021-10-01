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
""" Profiler with optional synchronization of JAX ant Torch if installed.

Example:
>>> import profiler from labtools
... profiler.start('train', 'fetch_batch')
... for batch in batch_iter:
...    profiler.end('fetch_batch')
...    # forward pass
...    with profile_kv('forward'):
...      image_features = model.encode_image(images)
...    profiler.start('fetch_batch')
... profiler.end('train')
... profiler.log_profiles()
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import time
from typing import Callable, Optional, Union

from absl import app
from absl import flags
from absl import logging

from labtools._src.util import maybe_import

Number = Union[int, float]

FLAGS = flags.FLAGS
flags.DEFINE_enum('labtools_profiling',
                  'disabled', ['disabled', 'enabled', 'strict'],
                  'Profiling mode.',
                  module_name='labtools')


class AverageMeter(object):
  """Computes and stores the average and current value"""

  __slots__ = ['val', 'avg', 'sum', 'count']

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val: Number, n: Number = 1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def synchronize_fn(fn: Callable,
                   sync_in: bool = True,
                   sync_out: bool = False) -> Callable:

  def _jax_blocker():
    if jax is not None:
      return jax.jit(jax.device_put)(0.0).block_until_ready()

  sync_fns = []
  torch = maybe_import('torch')
  jax = maybe_import('jax')
  if torch is not None:
    sync_fns.append(torch.cuda.synchronize)
  if jax is not None:
    sync_fns.append(_jax_blocker)

  sync_fn = lambda: [f() for f in sync_fns]
  sync_fn_in = sync_fn if sync_in else lambda: []
  sync_fn_out = sync_fn if sync_out else lambda: []

  @wraps(fn)
  def _synchronize_fn(*args, **kwargs):
    sync_fn_in()
    out = fn(*args, **kwargs)
    sync_fn_out()
    return out

  return _synchronize_fn


class Singleton(type):
  _instances = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]


class Profiler(metaclass=Singleton):
  """ Provides a profiler with optional synchronzition """
  _mode: str = None
  start: Callable[[str], None]
  end: Callable[[str], None]

  def __init__(self, name='root', mode=None):

    self._incontext = None
    self._enabled = False
    self._default_name = name

    # initialize
    self.reset_timers()

    # init once app initialized
    if name == 'root' and mode is None:
      app.call_after_init(lambda: self.complete_absl_config(flags))
    else:
      self.toggle(mode)

  def reset_timers(self):
    self._prevtimes = defaultdict(time.time)
    self._starttimes = defaultdict(time.time)
    self._counters = defaultdict(AverageMeter)

  def complete_absl_config(self, absl_flags):
    self.toggle(absl_flags.FLAGS.labtools_profiling)

  @property
  def prev(self):
    return self._prev

  @property
  def mode(self):
    return self._mode

  def toggle(self, mode: str):
    # noop
    if mode == self._mode:
      return
    self._mode = mode
    self._enabled = True
    if mode == 'strict':
      logging.info('Profiler set to strict mode, will synchronize Torch/Jax.')
      self.start = synchronize_fn(self.__start)
      self.end = synchronize_fn(self.__end)
    elif mode == 'enabled':
      logging.warning('Profiler not in strict mode, timings may be inaccurate.')
      self.start = self.__start
      self.end = self.__end
    elif mode == 'disabled':
      self.start = self.__passthrough
      self.end = self.__passthrough
      self._enabled = False
    else:
      raise ValueError('Mode %s not a valid profiling mode.' % mode)

  def enable(self, strict: bool = True):
    self.toggle(mode='strict' if strict else 'enabled')

  def disable(self):
    self.toggle(mode='disabled')

  def __start(self, *names: Optional[list[str]]):
    for name in names:
      name = name or self._default_name
      self._starttimes[name] = time.time()

  def __end(self, *names: Optional[list[str]]):
    curr = time.time()
    for name in names:
      elapsed = curr - self._starttimes[name]
      name = name or self._default_name
      # this might've been the first use, if so do count it
      if elapsed > 0:
        self._counters[name].update(elapsed)
      else:
        logging.warning(
            'Attemting to call profiler.end() on uninitialized timer.')

  def __str__(self):
    out = f'Profiler results ({self._default_name})\n'
    if len(self._counters) > 0:
      tabulate = maybe_import('tabulate')
      if tabulate:
        out += tabulate(self._counters, headers='keys')
      else:
        out += str(self._counters)
    return out

  def log_profiles(self, force: bool = False):
    if self._enabled or force:
      logging.info(str(self))

  def __passthrough(self, *_: str):
    return


# Create default profiler (singleton)
profiler = Profiler()


@contextmanager
def profile_kv(scopename):
  """ Provides a context manager for profiling a scope. """
  profiler.start(scopename)
  try:
    yield
  finally:
    profiler.end(scopename)


def profile(fn, n=None):
  """ Provides a decorator to profile a funcion

  Args:
    fn: function to decorate
    n: Name of the function. If not provided will use the name of the function.

  Example:
  >>> @profile
  ... def bar():
  ...   return 'hi'
  """
  scopename = n or fn.__name__

  @wraps(fn)
  def wrapped_fn(*args, **kwargs):
    profiler.start(scopename)
    out = fn(*args, **kwargs)
    profiler.end(scopename)
    return out

  return wrapped_fn
