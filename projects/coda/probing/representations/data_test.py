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
from datasets import Dataset
import numpy as np

from probing.representations.data import jax_multi_iterator


class TestData(absltest.TestCase):

  def test_dataset_compilation(self):
    """ compilation for the same number of jobs on a different inner run should
      not break the compilation.
    """
    ds = Dataset.from_dict({
        'hidden_states': np.random.rand(400, 40, 40),
        'label': np.random.rand(400, 11),
    })
    get_mi = jax_multi_iterator(ds, 32, True)

    jobs = [
        {
            'samples': 100,
            'indices': np.random.randint(0, 10, 100),
            'seed': 1
        },
        {
            'samples': 100,
            'indices': np.random.randint(0, 39, 100),
            'seed': 1
        },
        {
            'samples': 100,
            'indices': np.random.randint(0, 18, 100),
            'seed': 2
        },
    ]

    for _ in get_mi(jobs):
      break
      ...

    jobs = [
        {
            'samples': 100,
            'indices': np.random.randint(0, 80, 100),
            'seed': 1
        },
        {
            'samples': 100,
            'indices': np.random.randint(0, 80, 100),
            'seed': 1
        },
        {
            'samples': 100,
            'indices': np.random.randint(0, 80, 100),
            'seed': 1
        },
    ]

    for _ in get_mi(jobs):
      break
      ...


if __name__ == '__main__':
  absltest.main()
