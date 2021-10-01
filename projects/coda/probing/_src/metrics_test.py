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
""" Tests for the core evaluation metrics  """

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import torch

from probing._src.metrics import compute_metrics


class TestMetrics(parameterized.TestCase):
  """Test Core Metrics """

  @parameterized.named_parameters(
      ('list', lambda x: x),
      ('numpy', np.asarray),
      ('jax', jnp.asarray),
      ('torch', torch.tensor),
  )
  def test_compute_metrics(self, constructor):
    """ Test that metrics work independent of of input data format. """
    x = [2, 12, 4, 8]
    preds = constructor(x)
    targets = constructor(x)

    expected = {
        'spearman': 1,
        'spearman_p': 0,
        'kendalls_tau': 1,
        'corr_avg': 1,
        'acc': 1,
        'jensenshannon_dist': 0.0,
        'jensenshannon_div': 0.0,
    }
    result = compute_metrics(preds, targets)
    result.pop('kendalls_tau_p')  # ?
    self.assertEqual(expected, result)


if __name__ == '__main__':
  absltest.main()
