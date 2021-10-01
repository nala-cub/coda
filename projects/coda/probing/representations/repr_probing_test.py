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

from absl.testing import absltest
from absl.testing import parameterized

from probing.configs import get_configs
from probing.representations.repr_probing import repr_probing


class TestReprProbing(parameterized.TestCase):

  @parameterized.named_parameters(*get_configs(None, with_names=True))
  def test_repr_probing_correct_args(self, config):
    """ test that each config provides the correct arguments """

    with self.assertRaises(FileNotFoundError):
      repr_probing(repr_ds=None, **config.repr)


if __name__ == '__main__':
  absltest.main()
