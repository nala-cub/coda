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
""" Provides common io utilities """

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Union

from absl import logging
import cytoolz.curried as T
from rules_python.python.runfiles import runfiles

from labtools._src.util import BestEffortJSONEncoder
from labtools._src.util import CustomJSONEncoder
from labtools._src.util import maybe_import
from labtools._src.util import require

# try imports
yaml = maybe_import('yaml')


@require('yaml')
def load_and_check_yml(path: Union[str, Path], *loadkeys: list[str]):
  """ Loads and extract the specified key(s) from a yml file

    Args:
      path: Path to the yml file
      *loadkeys: Keys to load from the config file. These can either be shallow
        keys or deep addresses seperated as '/'. In the latter case, the file
        will be flattened to find the objects recursively.

    Returns:
      A list of Objects cooresponding to each key in loadkeys. Note that keys
      not present will have a value of None
  """
  with open(path, 'r', encoding='utf-8') as f:
    loaded = yaml.safe_load(f)

  ret = list(T.map(lambda x: T.get_in(x.split('/'), loaded), loadkeys))
  return ret


def dump_jsonl(path: Union[Path, str],
               data: list[dict[str, Any]],
               relaxed: bool = True) -> None:
  """ Dump to jsonl.
  Args:
    path: Path to the jsonl file.
    data: object to dump.
    relaxed: predicate indicating whether to throw an error when part of the
      data cannot be encoded using CustomJSONEncoder.
  """
  path = Path(path)
  path.parent.mkdir(exist_ok=True, parents=True)
  encoder_cls = BestEffortJSONEncoder if relaxed else CustomJSONEncoder
  # maybe it's a datframe
  if str(type(data)) == "<class 'pandas.core.frame.DataFrame'>":
    data.to_json(  # pytype: disable=attribute-error
        path, orient='records', lines=True)
  else:
    with open(path, 'w', encoding='utf-8') as f:
      for obj in data:
        f.write(json.dumps(obj, cls=encoder_cls) + '\n')


def load_jsonl(path: Union[Path, str]) -> Generator[Dict[str, Any], None, None]:
  """ Load from jsonl.

  Args:
    path: Path to the jsonl file

  """
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      yield json.loads(line)


def dump_json(path: Union[Path, str],
              data: dict[str, Any],
              relaxed: bool = True,
              indent=4) -> None:
  """ Dump to json.
  Args:
    path: Path to the jsonl file.
    data: object to dump.
    relaxed: predicate indicating whether to throw an error when part of the
      data cannot be encoded using CustomJSONEncoder.
      indent: json indentation.
  """
  path = Path(path)
  path.parent.mkdir(exist_ok=True, parents=True)
  encoder_cls = BestEffortJSONEncoder if relaxed else CustomJSONEncoder
  path.write_text(json.dumps(data, cls=encoder_cls, indent=indent))


def maybe_rlocation(path: str) -> str:
  r = runfiles.Create()
  resolved_path = r.Rlocation(path)
  if resolved_path is None:
    logging.warning(
        'failed to resolve %s in runfiles tree. returning the original path.',
        path)
    resolved_path = path
  return resolved_path
