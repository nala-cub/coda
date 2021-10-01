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

import copy
from typing import List, Tuple, TypedDict

from absl import logging
from chex import assert_max_traces
from datasets import Dataset
from datasets import DatasetDict
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import toolz.curried as T
import tree


class Job(TypedDict):
  seed: int
  indices: np.ndarray
  samples: int
  num_objects: int


def generate_jobs(  # pytype: disable=bad-return-type
    ds: DatasetDict,
    nb_seeds: int,
    nb_points: int,
    seed: int,
) -> List[Job]:
  """ Generate training jobs

  Args:
    ds: dataset containing representations
    nb_seeds: number of random seeds to use per model/point configuration
    nb_points: number of points, i.e the number distinct splits to evaluate in
      the dataset with increasing number of objects.
    seed: random seed used for generating the object groups.

  Returns:
    jobs: a list of jobs describing the models to be trained.
  """

  # generate a map of class_id -> indices.
  class_ids = ds.unique('class_id')
  index_map = {_id: [] for _id in class_ids}

  def get_index_mapping(class_id, idx):
    index_map[class_id].append(idx)

  ds.map(get_index_mapping, with_indices=True, input_columns=['class_id'])

  nobjects = len(class_ids)
  points = np.ceil(np.logspace(0, np.log10(nobjects), nb_points)).astype(int)
  rng = np.random.default_rng(seed)

  jobs = []
  # each increasing point adds more data (but keeps the same initial data)
  for seed_idx in range(nb_seeds):
    seed_cls_ids = rng.permutation(class_ids)
    for point in points:
      job_cls_ids = seed_cls_ids[:point]
      job_indices = np.concatenate(
          [index_map[cls_id] for cls_id in job_cls_ids])

      job = {
          'seed': seed + seed_idx,
          'indices': job_indices,
          'samples': len(job_indices),
          'num_objects': point
      }
      jobs.append(job)
  # jobs: [{seed: seed_idx, indices: <Subset_Size>, nobjects: N}]
  logging.info('Points: %s, %d', points, len(ds.unique('class_id')))
  obj_points = np.unique(list(T.map(T.get('num_objects'), jobs)))
  sam_points = np.unique(list(T.map(T.get('samples'), jobs)))
  logging.debug('Object Ns: %s', obj_points)
  logging.debug('Sample Ns: %s', sam_points)
  jobs = _pad_jobs(jobs)
  return jobs


def _pad_jobs(jobs: List[Job]):
  """ pad the jobs s.t. they all have the same number of indices

    We will slice only up to the max samples for a given object, so that is not
    an issue. this ensures we don't recompile each new iter job set - which is
    much worse performance-wise..
  """
  jobs = copy.deepcopy(jobs)
  # pad the indices so we can vmap sampling easier (otherwise it'd be ragged)
  max_samples = max([job['samples'] for job in jobs])

  for job in jobs:
    job['indices'] = np.pad(job['indices'], [(0, max_samples - job['samples'])],
                            'empty')
  # now we can stack the jobs as jax arrays
  # {indices: <njobs,max_samples>}
  return jobs


import time


def _jax_multi_iterator(
    ds,
    ds_fits_in_vram: bool = True,
    max_traces: int = 1,
):
  # small(ish) dataset. throw it on all vram.
  # this is really fast, but may have it's limits in terms of models.
  if ds_fits_in_vram:
    logging.info('setting format to jax.')
    # this seems to help with load time?? like minutes -> seconds..
    ds.set_format('jax', columns=['hidden_states', 'label'])
    logging.info('loading dataset to vram.')
    ts = time.time()
    data_x = jnp.asarray(ds['hidden_states'])
    data_y = jnp.asarray(ds['label'])
    logging.info('loaded dataset in %f seconds.', time.time() - ts)
  else:
    raise NotImplementedError

  print(data_x.shape, data_y.shape)

  @jax.jit
  @assert_max_traces(n=max_traces)
  def get_example(job, i):
    # generate a new dataset seed to prevent overlap of consecutive seeds
    dataset_seed = jr.split(jr.PRNGKey(job['seed']), 1)[0][0]

    point_seed = dataset_seed + i
    indices_index = jr.randint(jr.PRNGKey(point_seed),
                               shape=(),
                               minval=0,
                               maxval=job['samples'])

    point_index = lax.dynamic_index_in_dim(job['indices'],
                                           indices_index,
                                           keepdims=False)

    x_i = lax.dynamic_index_in_dim(data_x, point_index, keepdims=False)
    y_i = lax.dynamic_index_in_dim(data_y, point_index, keepdims=False)
    return {'input': x_i, 'label': y_i}

  get_batch = jax.vmap(get_example, in_axes=(None, 0))
  get_multibatch = jax.vmap(get_batch, in_axes=(0, None))

  return get_multibatch


@T.curry
def iterate_multibatch(multibatch_fn, jobs, batch_size: int):
  i = 0
  while True:
    indices = jnp.arange(i, i + batch_size, dtype=jnp.int32)
    yield multibatch_fn(jobs, indices)
    i += batch_size


def jax_multi_iterator(
    ds: Dataset,
    batch_size: int,
    ds_fits_in_vram: bool = True,
    max_traces: int = 1,
):
  # get the multibatch iter, which should be shared across different jobs.
  get_multibatch = _jax_multi_iterator(ds, ds_fits_in_vram, max_traces)
  # return a curried func that has the shared function and batch size. This just
  # needs "jobs"
  loader_iter = iterate_multibatch(get_multibatch, batch_size=batch_size)

  def _get_loader_iter_for_jobs(jobs):
    jobs = tree.map_structure(lambda *x: jnp.stack(x), *jobs)
    return loader_iter(jobs)

  return _get_loader_iter_for_jobs


def dummy_dataset(
    key_seq: hk.PRNGSequence,
    batch_size: int,
    input_shape: Tuple[int, ...],
    n_classes: int,
    n_models: int,
    **_,
):
  """Loads the dataset as a generator of batches."""

  if n_classes <= n_models:
    logging.fatal('Cannot have less classes than models.')

  def data_iter():
    while True:
      x = jr.normal(next(key_seq), [n_models, batch_size, *input_shape])
      y = jnp.tile(jnp.arange(0, n_models).reshape(-1, 1), (1, batch_size))
      oh_y = jax.nn.one_hot(y, num_classes=n_classes)
      yield {'input': x, 'label': oh_y}

  return data_iter()
