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
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from labtools import param_product
from probing._src.metrics import jensenshannon_div as scipy_jensenshannon_div
from probing.representations.data import dummy_dataset
from probing.representations.models import build_models
from probing.representations.models import jensenshannon_div


class TestMLP(parameterized.TestCase):

  @param_product(
      batch_size=[1, 4],
      input_shape=[(1,), (512,), (512, 9)],
      n_classes=[101],
      num_models=[1, 10, 100],
  )
  def test_mdl_vmap(self, batch_size, input_shape, n_classes, num_models):
    """ Test convergence with a set of MLPs
      We vmap each model and use a dummy dataset where x = N(0, 1) and Y is the
      model position. This primarily checks that each model is being optimized
      independently in parallel.
    """
    # run as main
    rng = jr.PRNGKey(0)
    init_fn, update_fn, metrics_fn = build_models(input_shape, (512, 512),
                                                  batch_size, n_classes, 1e-3,
                                                  True)

    # Create data RNGS
    rngs = jr.split(rng, num_models)
    rngs, init_rngs, data_rngs = zip(*[jr.split(rng, 3) for rng in rngs])
    params, opt_state = init_fn(init_rngs)

    rngs = jnp.asarray(rngs)
    key_seq = hk.PRNGSequence(data_rngs[0])
    batch_iter = dummy_dataset(key_seq, batch_size, input_shape, n_classes,
                               num_models)
    for _ in range(20):
      batch = next(batch_iter)
      params, opt_state, metrics, rngs = update_fn(params, rngs, opt_state,
                                                   batch)

    # test predictions on a new batch?
    # preds, metrics = metrics_fn(params, next(batch_iter))
    metrics = metrics_fn(params, batch)
    pred_label = jnp.argmax(metrics['preds'], axis=-1)

    np.testing.assert_equal(pred_label[:, 0], np.arange(num_models))

  def test_jsd_jax(self):
    rng = jr.PRNGKey(0)
    shape = (8, 64)
    rng, xrng, yrng = jr.split(rng, 3)
    x = jr.normal(xrng, shape)
    y = jr.normal(yrng, shape)
    ref = scipy_jensenshannon_div(x, y, axis=-1)

    res = jensenshannon_div(x, y, axis=-1)
    np.testing.assert_equal(res, ref)


if __name__ == '__main__':
  absltest.main(failfast=True)
