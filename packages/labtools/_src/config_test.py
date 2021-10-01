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
""" Provides tests for `labtools._src.config` """

from absl.testing import absltest
from absl.testing import parameterized
from ml_collections import ConfigDict
from ml_collections import FrozenConfigDict

from labtools._src.config import frozen


class TestFrozen(parameterized.TestCase):
  """ Provides tests for the @frozen decorator """

  def test_frozen(self):
    config = ConfigDict()

    @frozen
    def basic_config_fn():
      return config

    frozen_config = basic_config_fn()

    self.assertEqual(frozen_config, FrozenConfigDict(config))


if __name__ == '__main__':
  absltest.main()
