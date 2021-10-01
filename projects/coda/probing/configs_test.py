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
""" Config tests """

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import tree

from probing.configs import get_configs
from probing.configs import get_known_config_builders

KNOWN_CONFIGS = get_known_config_builders()

SMALL_MODELS = [
    'randinit-roberta',
    'randinit-clip',
    'clip-vitb32',
    'clip-rn50',
    'gpt2',
    'roberta-base',
    'albert-base-v1',
    'albert-base-v2',
]


class TestConfigs(parameterized.TestCase):
  """ Test configurations """

  @parameterized.named_parameters(*KNOWN_CONFIGS)
  def test_config_builds(self, config_fn):
    _ = config_fn()

  def test_configs_have_same_structure(self):
    configs = get_configs(None)

    if len(configs) > 0:
      for config in configs:
        tree.assert_same_structure(configs[0].to_dict(), config.to_dict())

  @parameterized.named_parameters(
      ('special/all', 'special/all', [n for n, _ in KNOWN_CONFIGS]),
      ('default', None, [n for n, _ in KNOWN_CONFIGS]),
      ('small', 'special/small', SMALL_MODELS),
      ('gpt2-all', 'gpt2', ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']))
  def test_get_configs(self, query, expected):
    """ Test regex matches"""
    configs = get_known_config_builders(query, True)
    config_names = [c[0] for c in configs]
    self.assertEqual(config_names, expected)


if __name__ == '__main__':
  absltest.main()
