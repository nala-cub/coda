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
""" integration tests for the zeroshot pipeline """

from absl.testing import absltest

from probing.configs import get_configs
from probing.zeroshot.zeroshot import _zeroshot_pipeline


class TestZeroshotPipeline(absltest.TestCase):
  prefix = 'com_github_corypaik_coda/projects/coda/'

  def test_run_no_configs(self):
    """ test with no configs (no results)"""
    result = _zeroshot_pipeline(
        configs=[],
        ngram_gbc_path=self.prefix + 'data/ngram-counts-gbc.csv',
    )
    self.assertIsNone(result)

  def test_zeroshot_pipeline_smoke(self, config_str='special/small'):
    """ test with no configs (no results)"""
    result = _zeroshot_pipeline(
        configs=get_configs(config_str, with_names=True),
        ngram_gbc_path=self.prefix + 'data/ngram-counts-gbc.csv',
        max_examples=8,
    )
    self.assertIsNotNone(result)


if __name__ == '__main__':
  absltest.main()
