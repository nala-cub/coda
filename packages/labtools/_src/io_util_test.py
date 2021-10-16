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
""" Provides tests for `labtools._src.io_util` """
from absl.testing import absltest
from absl.testing import parameterized

from labtools._src.io_util import _download_file
from labtools._src.io_util import download_files
from labtools._src.io_util import dump_jsonl


class JsonlTest(parameterized.TestCase):

  def test_dump_jsonl(self):
    data = [{'a': 0}, {'b': 1}, {'me': 'the'}]
    dump_jsonl(self.create_tempfile(), data)


class DownloadFilesTest(parameterized.TestCase):

  def test__download_file(self):
    p = self.create_tempfile()
    url = 'https://dummyimage.com/600x400/000/fff'
    res = _download_file((url, p))
    # Check success.
    self.assertEqual(res, (1, None))

  def test_download_files(self):
    download_dir = self.create_tempdir()
    n = 10
    tasks = [{
        'filename': str(i),
        'url': url
    } for i, url in enumerate(['https://dummyimage.com/600x400/000/fff'] * n)]

    num_completed = download_files(tasks, download_dir)
    self.assertEqual(num_completed, n)

    # Default clobber=False should now download 0 files.
    num_completed = download_files(tasks, download_dir)
    self.assertEqual(num_completed, 0)


if __name__ == '__main__':
  absltest.main()
