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
""" LabTools Public API

Python package `labtools` provides various utilities for use cases such as
experiment managers with error handling and cache cleanup, profiling,
working with tfrecords, converting arbitrary array-type objects to lists, etc.

<!--
isort:skip_file
-->
"""


def _is_labtools_testing_stub():
  from os import environ  # pylint: disable=import-outside-toplevel
  return 'LABTOOLS_TESTING_INTERNAL' in environ


if not _is_labtools_testing_stub():
  from labtools._src.util import is_installed
  from labtools._src.util import maybe_import
  from labtools._src.util import require
  from labtools._src.util import catch_exp_failures
  from labtools._src.util import topylist
  from labtools._src.util import compute_obj_hash
  from labtools._src.util import CustomJSONEncoder
  from labtools._src.util import BestEffortJSONEncoder
  from labtools._src.util import ensure_listlike
  from labtools._src.util import safe_zip
  from labtools._src.util import safe_map
  from labtools._src.util import unzip
  from labtools._src.util import flatten_dict
  from labtools._src.util import split_by_keys
  from labtools._src.util import get_differences
  from labtools._src.util import param_product
  from labtools._src.util import cleanup_cache_files
  from labtools._src.util import tolist

  from labtools._src.io_util import dump_json
  from labtools._src.io_util import dump_jsonl
  from labtools._src.io_util import load_jsonl
  from labtools._src.io_util import load_and_check_yml
  from labtools._src.io_util import maybe_rlocation

  from labtools._src.config import setup_jupyter_env
  from labtools._src.config import frozen
  from labtools._src.config import configure_logging

  from labtools._src.profiling import Profiler
  from labtools._src.profiling import profiler
  from labtools._src.profiling import profile
  from labtools._src.profiling import profile_kv

  from labtools._src.huggingface import hf_get_fwd_columns
  from labtools._src.huggingface import hf_one_to_many

__all__ = (
    'is_installed',
    'maybe_import',
    'require',
    'catch_exp_failures',
    'cleanup_cache_files',
    'topylist',
    'tolist',
    'compute_obj_hash',
    'CustomJSONEncoder',
    'BestEffortJSONEncoder',
    'ensure_listlike',
    'safe_zip',
    'safe_map',
    'unzip',
    'frozen',
    'flatten_dict',
    'split_by_keys',
    'get_differences',
    'dump_json',
    'dump_jsonl',
    'load_jsonl',
    'load_and_check_yml',
    'setup_jupyter_env',
    'configure_logging',
    'maybe_rlocation',
    'Profiler',
    'profiler',
    'profile',
    'profile_kv',
    'hf_get_fwd_columns',
    'hf_one_to_many',
    'param_product',
)
