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
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from absl import logging
import cytoolz.curried as T
import requests
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


def _download_file(
    task: Tuple[str, str]) -> Union[Tuple[int, str], Tuple[int, None]]:
  """ Download a single file to a specified path

  Args:
    task: A tuple containing `(url, filepath)`, which specifies where to
      download the file. This function will not create directories, so the
      parent of `filepath` should exist prior to execution.

  Returns:
    A tuple containing the download status and an error string.
      (1, None) for successful downloads
      (0, errmsg) for unsuccessful downloads

  """
  url, fpath = task
  logging.debug('Downloading %s to %s', url, fpath)
  try:
    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
      with open(fpath, 'wb') as f:
        for data in r:
          f.write(data)
  except:
    logging.exception('Failed to download %s', url)
    # cleanup
    Path(fpath).unlink(missing_ok=True)
    return 0, 'error -failed to dl or write'
  return 1, None


def download_files(tasks: List[Dict[str, Any]],
                   download_dir: Optional[Union[str, Path]] = None,
                   num_threads: Optional[int] = None,
                   clobber: bool = False,
                   filename_key: str = 'filename',
                   url_key: str = 'url') -> int:
  """ Download a list of files, optionally overwriting existing files.

  Args:
    tasks: List containing download tasks. Each task should have at minimum
      `filename_key` and `url_key`. Other entries will be ignored
    download_dir: Directory to download results, if provided.
    num_threads: Number of downloader threads to use. Defaults to the number of
      supported threads (as reported by `multiprocessing.cpu_count()`).
    clobber: Predicate indicating that existing files should be deleted. The
      default of `False` means that existing files will be skipped.
    filename_key: Key to use as the filename in `tasks`
    url_key: Key to use as the url in `tasks`

  Returns:
    The number of files successfully downloaded. Note that this does not
    include files which were skipped or failed to download.

  """
  og_n_tasks = len(tasks)

  if download_dir:
    Path(download_dir).mkdir(exist_ok=True, parents=True)
    tasks = map(
        lambda task: {
            **task, filename_key: Path(download_dir, task[filename_key])
        }, tasks)

  # Maybe filter
  if not clobber:
    tasks = T.filter(lambda task: not task[filename_key].is_file(), tasks)
    tasks = list(tasks)
    n_skipped = og_n_tasks - len(tasks)
    if n_skipped:
      logging.info('Skipping %d existing files.', n_skipped)

  num_threads = num_threads or mp.cpu_count()
  logging.info('Downloading %d files w/ %d threads', len(tasks), num_threads)

  num_tasks = len(tasks)
  # Map -> List[Tuple[url, filename]]
  tasks = map(T.get([url_key, filename_key]), tasks)

  num_completed = 0
  with ThreadPool(num_threads) as p:
    results = p.imap_unordered(_download_file, tasks)
    for s, r in results:
      if r:
        logging.error('download %s', r)
      num_completed += s

  logging.info('Successfully downloaded %d/%d files.', num_completed, num_tasks)
  return num_completed
