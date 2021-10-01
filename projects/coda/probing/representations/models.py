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

from typing import Any, Callable, List, Mapping, Optional, Tuple, TypedDict

from absl import flags
from absl import logging
import haiku as hk
from icecream import ic
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import rlax
import toolz.curried as T
import tree

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Batch = Mapping[str, np.ndarray]
Metrics = Mapping[str, np.ndarray]
UpdateFun = Callable[[hk.Params, OptState, Batch], Tuple[hk.Params, OptState,
                                                         Metrics]]
MetricsFun = Callable[[hk.Params, Batch], Tuple[hk.Params, Metrics]]
# for the purposes of our iterators, this is the same.
Dataset = Batch

RNG = Any


class Batch(TypedDict):
  input: jnp.ndarray
  label: jnp.ndarray


def build_models(
    input_shape: Tuple[int, ...],
    hidden_sizes: Tuple[int, ...],
    batch_size: int,
    n_classes: int,
    learning_rate: float = 1e-3,
    add_vmaps: bool = False,
) -> Tuple[Callable, UpdateFun, MetricsFun]:
  """ Build a set of models to train in parallel.

  Notes:
    - The result of this function is a set of stacked parameter and optimizer
      states with cooresponding vmapped update and metric functions.
    - While each model will be *initialized* with a different set of parameters,
      they will all have the same hyperparameters.
    - This is designed to work for testing multiple seeds of the same model, or
      with different input data per model.

  Args:
    batch_size: batch_size for a single model
    input_shape: input shape for a single model
    n_classes: number of classes

  Returns:
    # init_fn: curried initialization function, which takes as input PRNGs and
      creates params / opt states to be used with update_fn and metrics_fn
    update_fn: possibly vmapped update function.
    metrics_fn: possibly vmapped metrics function.
  """

  # build the forward function for *all* model params
  forward_fn = build_forward_fn(output_sizes=[*hidden_sizes, n_classes])
  network = hk.transform(forward_fn)

  # optimizer
  opt = optax.adam(learning_rate)

  # create update rule mapped across all models
  update_fn = build_update_fn(network.apply, opt.update, n_classes)

  # same for metrics rule.
  metrics_fn = build_metrics_fn(network.apply, n_classes)

  if add_vmaps:
    update_fn = jax.vmap(update_fn)
    metrics_fn = jax.vmap(metrics_fn)

  # parital for init
  init_fn = initialize_models(network, opt, batch_size, input_shape)

  return init_fn, update_fn, metrics_fn


@T.curry
def initialize_models(
    network,
    opt,
    batch_size: int,
    input_shape: Tuple[int, ...],
    init_rngs: List[RNG],
) -> Tuple[hk.Params, OptState]:
  """ Initialize a set of models to train in parallel.

  Args:
    network: transformed network
    opt: optimizer
    batch_size: per-model batch size.
    input_shape: input shape for the MLPs
    init_rngs: list of PRNG keys, one per model.

  Returns:
    params: A stacked set of parameters for use with vmapped functions.
    opt_state: A stacked optimizer state for use with vmap.
  """

  # initialize models and optimizer states.
  data = jnp.zeros([batch_size, *input_shape])
  params = [network.init(init_rng, data) for init_rng in init_rngs]
  opt_state = [opt.init(model_params) for model_params in params]

  # stack parameters and optimizer states to compute with in parallel via vmap
  params = tree.map_structure(lambda *x: jnp.stack(x), *params)
  opt_state = tree.map_structure(lambda *x: jnp.stack(x), *opt_state)

  return params, opt_state


def build_forward_fn(*args, **kwargs):

  def forward_fn(x: jnp.ndarray, dropout_rate: float = 0.0) -> jnp.ndarray:
    """Standard MLP probe."""
    rng = hk.next_rng_key()
    x = hk.Flatten()(x)
    x = hk.nets.MLP(*args, **kwargs)(x, dropout_rate=dropout_rate, rng=rng)
    return x

  return forward_fn


def build_update_fn(forward_fn, opt_update_fn, n_classes) -> UpdateFun:
  """Build the learning rule."""

  def update(params, rng, opt_state, batch: Batch):

    def loss_fn(_params, _batch):
      logits = forward_fn(_params, rng, _batch['input'], dropout_rate=0.5)
      labels = _batch['label']

      loss = optax.sigmoid_binary_cross_entropy(logits, labels)
      loss = jnp.mean(loss)

      l2_loss = 0.5 * sum(
          jnp.sum(jnp.square(p)) for p in jax.tree_leaves(_params))
      return loss + 1e-4 * l2_loss

    # Compute gradient and loss.
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    # Transform the gradients using the optimizer.
    updates, opt_state = opt_update_fn(grads, opt_state)
    # Update parameters.
    new_params = optax.apply_updates(params, updates)
    metrics = {'loss': loss}
    _, rng = jr.split(rng)
    return new_params, opt_state, metrics, rng

  return jax.jit(update)


@jax.jit
def rel_entr(x, y):
  """ Compute the relative entropy (same as scipy.special.rel_entr)

    rel_entr(x, y) = { x log(x/y) x>0,y>0
                     . 0          x=0,y>=0
                     . infty      otherwise
  """
  case_1 = jnp.logical_and(x > 0, y > 0)
  case_2 = jnp.logical_and(x == 0, y >= 0)
  rc1 = x * jnp.log(x / y)
  res = jnp.where(case_1, rc1, np.inf)
  res = jnp.where(case_2, 0., res)
  return res


def jensenshannon_div(p, q, *, axis=0, keepdims=False):
  """ Jensen Shannon Divergence.

  Based on scipy.distance.jensenshannon, without the sqrt.
  """
  p = p / jnp.sum(p, axis=axis, keepdims=True)
  q = q / jnp.sum(q, axis=axis, keepdims=True)
  m = (p + q) / 2.0
  left = rel_entr(p, m)
  right = rel_entr(q, m)
  left_sum = jnp.sum(left, axis=axis, keepdims=keepdims)
  right_sum = jnp.sum(right, axis=axis, keepdims=keepdims)
  js = left_sum + right_sum
  return js / 2.0


def build_metrics_fn(forward_fn, n_classes: int) -> MetricsFun:

  def metrics_fn(params: hk.Params, batch: Batch):
    """Evaluates the model at the given params/state."""
    labels = batch['label']
    logits = forward_fn(params, jr.PRNGKey(0), batch['input'])
    preds = jax.nn.softmax(logits)

    # metrics per example
    sbce_loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    # avg across labels
    sbce_loss = jnp.mean(sbce_loss, axis=-1)
    #
    log_loss = rlax.log_loss(preds, labels)
    log_loss = jnp.mean(log_loss, axis=-1)

    pred_label = jnp.argmax(preds, axis=-1)
    top_label = jnp.argmax(labels, axis=-1)
    top1acc = jnp.equal(pred_label, top_label)

    jsd = jensenshannon_div(preds, labels, axis=-1)

    # ->
    metrics = {
        'acc': top1acc,
        'sbce_loss': sbce_loss,
        'log_loss': log_loss,
        'preds': preds,
        'jensenshannon_div': jsd
    }
    return metrics

  return jax.jit(metrics_fn)


def build_val_iter(val_ds: Dataset, batch_size: Optional[int] = None):
  ds_size = len(val_ds)
  batch_size = batch_size or ds_size
  logging.info('Start val loop.')

  def data_iter():
    i = 0
    x = jnp.asarray(val_ds['hidden_states'][:])
    y = jnp.asarray(val_ds['label'][:])
    while i < ds_size:
      sli = slice(i, i + batch_size)
      batch = {'input': x[sli], 'label': y[sli]}
      yield batch
      i += batch_size

  return data_iter()


@T.curry
def evaluate(metrics_fn: MetricsFun, ds: Dataset, batch_size, params):
  metrics = []
  ds_iter = build_val_iter(ds, batch_size)
  for batch in ds_iter:
    batch_metrics = metrics_fn(params, batch)
    metrics.append(batch_metrics)
  # stack along the batch axis
  # metrics: {<name>: <num_jobs, num_examples, ...?> }
  metrics = tree.map_structure(lambda *x: jnp.concatenate(x, 1), *metrics)

  return metrics
