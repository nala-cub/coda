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
""" Representation Probing """

from typing import Optional, Tuple, Union

from absl import app
from absl import logging
from datasets import DatasetDict
from datasets import load_from_disk
from einops import rearrange
from einops import repeat
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from pandas import DataFrame
import toolz.curried as T
from tqdm import trange
import tree

from probing._src.configurable import configurable
from probing._src.constants import COLORS
from probing.representations import data
from probing.representations import models


@configurable
def repr_probing(  # pytype: disable=annotation-type-mismatch
    repr_ds: Optional[str] = None,
    preds_path: Optional[str] = None,
    results_path: Optional[str] = None,
    seed: int = 12345,
    nb_seeds: int = 5,
    nb_points: int = 10,
    batch_size: int = 64,
    n_training_steps: int = 4000,
    max_parallel: int = -1,
    log_freq: int = 0,
    max_batch_size: int = 1024,
    ds_fits_in_vram: bool = True,
    learning_rate: float = 1e-4,
    hidden_sizes: Tuple[int] = (512, 512),
    validation_split: str = 'validation',
) -> Tuple[DataFrame, DataFrame]:
  """Run representation probing.

  Depending on the representation size, we may need to do jobs in smaller
  batches.

  Args:
    seed: Random seed
    nb_seeds: Number of random seeds per point
    nb_points: Number of point to run along the curve
    batch_size: Batch size for each model.
    n_training_steps: Number of training steps.
    max_parallel: Maximum number of models that can be trained in parallel.
    log_freq: Logging frequency
    max_batch_size: Maximum batch size to use during evaluation.
    learning_rate: Learning rate
    hidden_sizes: Size of each hidden layer.
    repr_dataset: Directory containing a hf dataset with representations.
    preds: path to store predictions
    results: path to store results in
    ds_fits_in_vram: predicate indicating if the dataset fits in VRAM. This
      should only be set as a last resort, max_parallel is much faster.
    validation_split: split to use for calculating validation metrics. this
      should be `validattion` or `test`.
  """
  if not isinstance(repr_ds, DatasetDict):
    repr_ds = load_from_disk(repr_ds)

  if validation_split == 'train':
    raise ValueError(
        'validation split cannot be train, choose one of "validation" or "test".'
    )
  if validation_split == "test":
    logging.warning('received validation_split="test".')

  jobs = data.generate_jobs(
      repr_ds['train'],
      nb_seeds=nb_seeds,
      nb_points=nb_points,
      seed=seed,
  )
  # configure chex compile assertions
  chex_expect_num_compile = 1
  if len(jobs) % max_parallel != 0:
    logging.warning(
        'the # of jobs (%d) should be divisible by max_parallel (%d), otherwise'
        'jax will have to recompile every step for the last set of models.',
        len(jobs), max_parallel)
    chex_expect_num_compile = 2

  val_ds = repr_ds[validation_split]

  # Create RNGs
  # Initialise the model's parameters and the optimiser's state.
  # each initialization uses a different rng.
  n_models = len(jobs)
  rng = jr.PRNGKey(seed)
  rngs = jr.split(rng, n_models)
  rngs, init_rngs, data_rngs = zip(*[jr.split(rng, 3) for rng in rngs])

  train_ds = repr_ds['train']
  val_ds = repr_ds['test']
  # build models.
  # Depending on the representation size, we may need to do jobs in smaller
  #  batches, however, we will maintain the same functions throughout.
  #  only the parameter sets need get reset.
  input_shape = np.shape(train_ds[0]['hidden_states'])
  n_classes = len(train_ds[0]['label'])
  init_fn, update_fn, metrics_fn = models.build_models(
      input_shape,
      hidden_sizes,
      batch_size=batch_size,
      n_classes=n_classes,
      learning_rate=learning_rate)

  # create train iter
  train_iter = data.jax_multi_iterator(
      train_ds,
      batch_size,
      ds_fits_in_vram=ds_fits_in_vram,
      max_traces=chex_expect_num_compile,
  )
  # add vmaps
  update_fn = jax.vmap(update_fn)
  # validation function uses the same data for all models.
  valid_fn = jax.vmap(metrics_fn, in_axes=(0, None))
  evaluate = models.evaluate(valid_fn, val_ds, max_batch_size)

  # Create inner loop --->
  inner_loop = _repr_curve_inner(train_iter, init_fn, update_fn, evaluate,
                                 log_freq, n_training_steps)

  # zip up the rngs into jobs and partition s.t. < max_parallel
  inner_jobs = list(zip(jobs, rngs, init_rngs, data_rngs))

  if max_parallel > 0:
    inner_jobs = T.partition_all(max_parallel, inner_jobs)
  else:
    inner_jobs = [inner_jobs]

  records, preds = zip(*T.map(inner_loop, inner_jobs))

  df = _format_predictions(val_ds, preds, jobs)

  # store results
  results = _generate_results(records)
  df_result = pd.DataFrame.from_records(results)

  # maybe save to files
  if preds_path:
    df.to_csv(preds_path, index=False)
  if results_path:
    df_result.to_csv(results_path, index=False)

  return df, df_result


@T.curry
def _repr_curve_inner(
    train_iter,
    init_fn,
    update_fn,
    evaluate,
    log_freq,
    n_training_steps,
    jobset,
):
  # unpack the jobset
  jobs, rngs, init_rngs, data_rngs = zip(*jobset)

  # initialize
  params, opt_state = init_fn(init_rngs)
  train_iter = train_iter(jobs)

  rngs = jnp.asarray(rngs)

  # log_freq = 400

  # batch: <num_models, batch_size, *input_shape>
  train_pbar = trange(int(n_training_steps), desc='Training')
  for step in train_pbar:
    batch = next(train_iter)
    params, opt_state, up_metrics, rngs = update_fn(params, rngs, opt_state,
                                                    batch)
    if log_freq > 0 and (step + 1) % log_freq == 0:
      val_metrics = evaluate(params)
      train_pbar.set_postfix(
          avg_loss=jnp.mean(up_metrics['loss']),
          # best average jsd.
          val_jsd=jnp.min(jnp.mean(val_metrics['jensenshannon_div'], axis=-1)),
      )

  # validation
  val_metrics = evaluate(params)
  # avg across examples
  per_job_val_metrics = tree.map_structure(lambda x: jnp.mean(x, axis=1),
                                           val_metrics)

  results = _metrics_to_results(per_job_val_metrics, jobs, n_training_steps)
  results = list(results)
  return results, val_metrics


def _generate_results(records):
  """" Generates and saves results. """
  results = [r for resset in records for r in resset]
  return results


def _metrics_to_results(metrics, jobs, t):

  def make_result(item):
    i, job = item
    res = tree.map_structure(lambda x: x[i], metrics)
    res.update({
        'seed': job['seed'],
        'objects': job['num_objects'],
        'samples': job['samples'],
        't': t,
    })
    return res

  return T.map(make_result, enumerate(jobs))


def _format_predictions(ds, results, jobs) -> pd.DataFrame:
  """ Save predictions to a CSV

    cols:
      example_idx,object_idx,preds,seed,samples

    Args:
      ds: Dataset peds were made on
      preds <n_jobs,n_examples,n_classes>: Predictions for all jobs.
      jobs <n_jobs,2>: List of tuples counting <seed,point>
  """
  # chex.assert_equal_shape([ds.labels, preds[0]])
  # we may have done the work in jobsets, stack preds
  # preds = jnp.concatenate(preds)
  results = tree.map_structure(lambda *x: jnp.concatenate(x), *results)
  # shapes = tree.map_structure(lambda x: x.shape, results)

  results['class_id'] = repeat(np.asarray(ds['class_id']),
                               'num_ex -> n num_ex',
                               n=len(jobs))
  results['template_idx'] = repeat(np.asarray(ds['template_idx']),
                                   'num_ex -> n num_ex',
                                   n=len(jobs))

  # add job metadata
  for k in ('samples', 'num_objects', 'seed'):
    results[k] = repeat(np.asarray([x[k] for x in jobs]),
                        'n -> n num_ex',
                        num_ex=len(ds))
  # add colors
  results['objects'] = results.pop('num_objects')
  preds = results.pop('preds')
  for i, color in enumerate(COLORS):
    results[color] = preds[:, :, i]

  # should now be all <n_jobs,n_examples>
  # ic(tree.map_structure(lambda x: x.shape, results))
  # flatten all
  results = tree.map_structure(
      lambda x: rearrange(x, 'n_jobs nex -> (n_jobs nex)'), results)

  # -> csv
  df = pd.DataFrame(results)
  return df


def main(_):
  repr_probing()


if __name__ == '__main__':
  app.run(main)
