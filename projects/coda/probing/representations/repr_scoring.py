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

from typing import Optional

from absl import app
from absl import logging
import numpy as np
import pandas as pd

from probing._src.configurable import configurable
from probing._src.constants import COLORS
from probing._src.metrics import compute_metrics


def repr_scoring(
    preds: pd.DataFrame,
    groundtruth: pd.DataFrame,
    savepath: Optional[str] = None,
) -> pd.DataFrame:
  """
  Input format:
    meta:
        class_id,object_group,ngram,...
  """
  # expected size after aggregation
  # should be num_model * nseeds * nobjects
  unique_counts = preds.nunique()
  id_cols = ['objects', 'seed', 'class_id']
  expected_size = np.prod(unique_counts[id_cols])
  logging.info('Expected Size: %d', expected_size)
  preds = preds.set_index(id_cols)

  # group by class_id and select the row with best metrics
  # this is the best template for that example (upper bound)
  # best_preds = preds.sort_values('sbce_loss', ascending=True)
  best_preds = preds.sort_values('jensenshannon_div', ascending=True)
  best_preds = best_preds.groupby(id_cols).head(1)

  # length should be num_model * nseeds * nobjects
  if len(best_preds) != expected_size:
    logging.fatal('Expected %d rows. found %d', expected_size, len(best_preds))
  best_preds = best_preds.reset_index()
  # best_preds:
  #   name,objects,seed,class_id
  # where:
  #   name: name of the model
  #   objects: number of objects for this trial
  #   seed: seed for this trial
  #   class_id: class uid for the object.

  predictions = best_preds.reset_index()

  # melt both the groundtruth and preds so that we can align them and make level
  # columns in a pivot table. we also use this to add `object_group` to our preds.
  groundtruth = groundtruth.melt(
      id_vars=[
          'class_id',
          'object_group',
      ],
      value_vars=COLORS,
      var_name='color',
      value_name='groundtruth',
  )

  predictions = predictions.melt(
      id_vars=[
          'objects',
          'seed',
          'class_id',
          'template_idx',
      ],
      value_vars=COLORS,
      var_name='color',
      value_name='preds',
  )

  # join the predictions with groundtruth. Note that here groundtruth has only 1
  # entry per object, while predictions has many (1 for each job and template)
  # this is not a very strong validation - it really doesn't tell us if we just
  # lost a bunch of values or not
  results = predictions.merge(
      groundtruth,
      how='left',
      on=[
          'class_id',
          'color',
      ],
      validate='many_to_many',
  )
  # **** align **** |     left     |       right
  # class_id, color , object_group , objects , seed, template_idx

  # pivot the table back to get rows corresponding to examples
  #
  # **** align **** |     left     |       right                   |  groundtruth ...
  # class_id, color , object_group , objects , seed, template_idx, | red, orange,...
  results = results.pivot_table(
      index=[
          'objects',
          'seed',
          'object_group',
          'class_id',
          'template_idx',
      ],
      columns=['color'],
      values=['preds', 'groundtruth'],
  )

  # get the other prediction metrics
  def compute_row_metrics(row):
    preds = row[('preds',)].values
    targets = row[('groundtruth',)].values
    metrics = compute_metrics(preds=preds, targets=targets)
    return pd.Series(metrics)

  results = results.apply(compute_row_metrics, axis=1)
  results = results.reset_index()
  # add to preds
  best_preds = best_preds.merge(
      results,
      on=[
          'objects',
          'seed',
          'class_id',
          'template_idx',
      ],
      suffixes=('_jax', ''),
  )

  # Here we just average the best predictions from each object to get
  # averages for each model over the seeds. we already took the best template.
  best_preds = best_preds.groupby([
      'objects',
      'object_group',
      'class_id',
  ],
                                  as_index=False).mean()

  if savepath:
    best_preds.to_csv(savepath, index=False)

  return best_preds


@configurable('repr_scoring')
def repr_scoring_cmd(
    preds_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    savepath: Optional[str] = None,
) -> pd.DataFrame:
  """ Run zeroshot scoring

  Args:
    metadata_path: File containing dataset metadata.
    preds_path: File containing zeroshot preds for 1 model.
    savepath: File to save results in, if provided.
  """
  # compute the real metrics
  preds = pd.read_csv(preds_path)
  meta = pd.read_csv(metadata_path)

  return repr_scoring(preds, meta, savepath)


def main(_):
  repr_scoring_cmd()


if __name__ == '__main__':
  app.run(main)
