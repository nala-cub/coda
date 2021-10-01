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
""" Provides configuration utilities """
from __future__ import annotations

from functools import wraps
import logging as py_logging
import os
from pathlib import Path
import sys
from typing import Union
import warnings

from absl.flags import FLAGS
from absl import logging
from absl.logging.converter import absl_to_standard

from labtools._src.util import maybe_import
from labtools._src.util import require


def setup_jupyter_env(ensure_project_root: Union[None, str] = 'WORKSPACE',
                      max_parents: int = 1):
  """  Setup a jupyter notebook environment.

  Args:
    ensure_project_root: File name that indicates the desired root. When using
      bazel this is usually the WORKSPACE file. To disable this behavior, use
      ensure_project_root=None.
    max_parents: maximum number of parent directories to climb out of. This is a
      safegaurd against cd failures to find the specified file.

  Example:
    >>> import labtools
    ... from absl import logging
    ... labtools.setup_jupyter_env()
    ... logging.info('I work now!')
        12:00:00 ── INFO ▷ I work now!

  """

  py_logging.basicConfig(format='%(asctime)s ── %(levelname)s ▷ %(message)s',
                         datefmt='%H:%M:%S',
                         handlers=[py_logging.StreamHandler(sys.stdout)])
  logger = py_logging.getLogger('absl')
  logger.setLevel('INFO')

  # check for a workspace file
  if ensure_project_root:
    og_pwd = os.getcwd()
    pdepth = 0
    try:
      for pdepth in range(max_parents + 1):
        if Path(ensure_project_root).is_file():
          break
        os.chdir('..')
      else:
        logging.warning(
            'Failed to find %s in CWD at level=%d. PWD=%s, '
            'Original PWD=%s. Switching back to Original wd.',
            ensure_project_root, pdepth, os.getcwd(), og_pwd)
        # switch back as a safegaurd.
        os.chdir(og_pwd)
    except OSError:
      logging.exception(
          'Failed to find %s in CWD at level=%d. PWD=%s, '
          'Original PWD=%s. Switching back to Original wd.',
          ensure_project_root, pdepth, os.getcwd(), og_pwd)
      os.chdir(og_pwd)


@require('absl')
def configure_logging(third_party_offset: int = 0, **offsets):
  """ Configure logging formatters and levels

  Configures logging formatters and levels for absl and third party libraries.
  Default logging levels for external libraries:

    | --------------- | --------------------------------------------|
    | `transformers`  | FLAGS.verbosity - third_party_verbosity - 1 |
    | `datasets`      | FLAGS.verbosity - third_party_verbosity - 2 |

  Example:
    A verbosity of 0 (info) would result in a verbosity of warning for
    transformers and error for datasets.
  Args:
    third_party_offset: offset to add to all third party verbosity levels. For
      example, in a distributed configuration, one could use a value of `0` on
      the main process and `1` on all other processes.
    **offset: offsets for third party libraries
  """

  warnings.filterwarnings('ignore',
                          message='The given NumPy array is not writeable')

  # default third party offsets
  default_offsets = {'transformers': 0, 'datasets': 1}
  offsets = {**default_offsets, **offsets}

  third_party_verbosity = FLAGS.verbosity - third_party_offset

  for name, offset in offsets.items():
    logger = py_logging.getLogger(name)
    # clip verbosity
    verbosity = max(third_party_verbosity - offset, logging.FATAL)
    logger.setLevel(absl_to_standard(verbosity))
    # TODO: apply format
  # ic only in debug if installed.
  icecream = maybe_import('icecream')
  if icecream is not None:
    if FLAGS.verbosity < 1:
      icecream.ic.disable()
    else:
      icecream.ic.configureOutput(includeContext=True)


@require('ml_collections', as_arg=True)
def frozen(ml_collections, fn):
  """ Wrapper to freeze a configuration function.
    Args:
      fn (Callable[[], ConfigDict]): configuration function

    Returns:
      FrozenConfigDict
  """

  @wraps(fn)
  def wrapped(*args, **kwargs):
    return ml_collections.FrozenConfigDict(fn(*args, **kwargs))

  return wrapped
