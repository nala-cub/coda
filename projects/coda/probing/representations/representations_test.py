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
from probing.representations.representations import _repr_pipeline


class TestZeroshotPipeline(absltest.TestCase):

  def test_run_no_configs(self):
    """ test with no configs (no results)"""
    result = _repr_pipeline(configs=[])
    self.assertIsNone(result)

  def test_representation_pipeline_smoke(self, config_str='^gpt2$'):
    """ test with config - results """
    result = _repr_pipeline(configs=get_configs(config_str, with_names=True),
                            max_examples=8)
    self.assertIsNotNone(result)


if __name__ == '__main__':
  absltest.main()
