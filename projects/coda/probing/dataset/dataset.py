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

import os
from typing import Dict, List, Tuple, TypeVar

from absl import app
from absl import flags
from absl import logging
import datasets
import numpy as np
import pandas as pd
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric
from pyclustering.utils.metric import type_metric
from scipy.spatial.distance import jensenshannon
from scipy import stats
import toolz.curried as T
import tree
import yaml

from probing._src.config_util import maybe_rlocation
from probing._src.constants import COLORS
from probing.dataset._templates import recursive_format

flags.DEFINE_string(
    'coda_ds_export_dir', None,
    'Path to write the json exported CoDa dataset for huggingface upload'
    'This path is resolved relative to `BUILD_WORKSPACE_DIRECTORY`.')

FLAGS = flags.FLAGS

_T = TypeVar('_T')


def _filter_annotations(df, kt_threshold=0):
  """ filters annotations by

  given a dataframe containing all annotations for a given object, filters
  """
  # filter skipped
  df = df[df.included == True]
  metric_name = 'kendall_tau'
  num_filtered = 0
  total_num_submit = len(df)
  # this is iterative
  while True:
    # get the ground truth
    groundtruth = df[COLORS].mean(0)
    # we need the kt between each row and the groundtruth. We can only drop 1.
    df[metric_name] = df[COLORS].apply(lambda single_annot: stats.kendalltau(
        single_annot, groundtruth).correlation,
                                       axis=1)
    # check if metric is below threshold
    droprow = df.nsmallest(1, metric_name).iloc[0]
    if droprow[metric_name] >= kt_threshold:
      break
    # drop
    drop_worker_id = droprow.name[1]
    drop_obj_name = droprow['display_name']
    logging.debug('dropping annotation for "%s" by worker %s (%s = %.3f)',
                  drop_obj_name, drop_worker_id, metric_name,
                  droprow[metric_name])
    # set included to false (dropped)
    df.loc[droprow.name, 'included'] = False
    # filter dropped
    df = df[df.included == True]
    num_filtered += 1

  # check size matches
  if num_filtered == 0:
    pass
  if num_filtered != total_num_submit - len(df):
    logging.error('expected to drop %d annotations, but dropped %d.',
                  num_filtered, total_num_submit - len(df))
    raise RuntimeError('hi')
  else:
    logging.debug('dropped %d annotations (new_total=%d)', num_filtered,
                  len(df))

  return df


def create_dataset(
    template_path:
    str = 'com_github_corypaik_coda/projects/coda/data/coda/templates.yaml',
    objects_path:
    str = 'com_github_corypaik_coda/projects/coda/data/coda/objects.jsonl',
    annotations_path:
    str = 'com_github_corypaik_coda/projects/coda/data/coda/annotations.jsonl',
    seed_for_splits: int = 12345,
    seed_for_kmeans: int = 0,
) -> Tuple[datasets.DatasetDict, pd.DataFrame]:
  """ Prepares a dataset and saves it disk

  Args:
    metadata_path: File to save with metadata about each object.
    output_dataset_dir: Directory to save the dataset to disk.
  Returns:
    ds: dataset containing all formatted examples (train, val, test splits)
    meta: dataframe containing metadata about each object.
  """
  # maybe convert paths
  template_path = maybe_rlocation(template_path)
  objects_path = maybe_rlocation(objects_path)
  annotations_path = maybe_rlocation(annotations_path)

  # process annotations
  df = pd.read_json(annotations_path, orient='records', lines=True)
  # normalize
  # normalize
  df[COLORS] = df[COLORS].div(df[COLORS].sum(axis=1), 0)
  df = df.set_index(['class_id', 'worker_id'], verify_integrity=True)
  # apply a filter
  df = df.groupby('class_id', as_index=False).apply(_filter_annotations)
  df = df.reset_index()
  # average annotations
  df = df.groupby('class_id', as_index=False).mean()
  # kmeans for groupings.
  df = _get_object_groups(df, seed=seed_for_kmeans)

  # add template data. this also drops a few objects that we have annotations
  # for but are not included.
  tdf = pd.read_json(objects_path, orient='records', lines=True)
  df = df.merge(tdf, on='class_id', validate='one_to_one')
  df = df.sort_values('class_id')
  meta = df
  templates = _load_templates(template_path=template_path)

  # the real dataset: split groundtruth and filtered
  # gives us a dict for each split containing a list of objects (example form)
  split_objects = _generate_splits(df, seed=seed_for_splits)

  def _process_split(x: List[Dict[str, _T]]) -> Dict[str, List[_T]]:
    x = T.mapcat(_generate_examples_for_obj(templates=templates), x)
    x = list(x)
    x = {k: [el[k] for el in x] for k in x[0].keys()}
    return x

  # map each
  data = T.valmap(_process_split, split_objects)

  # metadata
  features = datasets.Features({
      'class_id':
          datasets.Value('string'),
      'display_name':
          datasets.Value('string'),
      'ngram':
          datasets.Value('string'),
      'label':
          datasets.Sequence(datasets.Value('float')),
      'object_group':
          datasets.ClassLabel(names=('Single', 'Multi', 'Any')),
      'text':
          datasets.Value('string'),
      'template_group':
          datasets.ClassLabel(names=('clip-imagenet', 'text-masked')),
      'template_idx':
          datasets.Value('int32')
  })

  # create dataset
  ds = datasets.DatasetDict(
      **{
          split: datasets.Dataset.from_dict(
              mapping=mapping,
              features=features,
              split=split,
          ) for split, mapping in data.items()
      })
  return ds, meta


def _load_templates(template_path):
  with open(template_path, 'r') as f:
    template_groups = yaml.safe_load(f.read())['templates']

  # flat templates.
  templates = []
  for template_group in template_groups:
    for template in template_group['templates']:
      templates.append({
          'text': template,
          'template_group': template_group['name']
      })

  return templates


@T.curry
def _generate_examples_for_obj(row, templates):
  # format are/is maps
  tvars = row.pop('vars')
  label = row.pop('label')
  label = [label[color] for color in COLORS]

  if tvars['object_pl']:
    tvars['object_areis_pl'] = '{object_pl} {areis_pl}'.format_map(tvars)
  if tvars['object_s']:
    tvars['object_areis_s'] = '{object_s} {areis_s}'.format_map(tvars)
  # filter nulls from templates
  tvars = {k: v for k, v in tvars.items() if v}
  # fill in templates
  examples = []
  templates = sorted(templates, key=lambda x: x['text'])
  for template_idx, template in enumerate(templates):
    try:
      data = recursive_format([template['text']], mapping=tvars)
      ex = {
          'text': data[0],
          'label': label,
          'template_group': template['template_group'],
          'template_idx': template_idx,
          **row
      }
      examples.append(ex)
    except:
      pass
  return examples


def _get_object_groups(df, seed):
  # we don't care about *which* colors, as we want to group on color types, so
  # we used sorted probabilities. throughout testing this gives consistent
  # clusters (in terms of size, order changes of course)
  probs = df[COLORS]
  sprobs = np.sort(probs)[:, ::-1]
  X = sprobs

  # Use jensensenshan distance to compare the distributions.
  metric = distance_metric(type_metric.USER_DEFINED, func=jensenshannon)

  # Prepare initial centers using K-Means++ method.
  initial_centers = kmeans_plusplus_initializer(
      data=X,
      amount_centers=3,
      random_state=seed,
  ).initialize()

  # create K-Means algorithm with specific distance metric
  kmeans_instance = kmeans(X, initial_centers, metric=metric)

  # run cluster analysis and obtain results
  kmeans_instance.process()
  y_pred = kmeans_instance.predict(X)

  idx_min = X[:, 0].argmin()
  idx_max = X[:, 0].argmax()
  df['object_group'] = y_pred

  # name the clusters
  cmap = {
      y_pred[idx_max]: 'Single',
      y_pred[idx_min]: 'Any',
  }
  cmap[({0, 1, 2} - set(cmap.keys())).pop()] = 'Multi'
  df['object_group'] = df['object_group'].replace(cmap)

  return df


def _generate_splits(df, seed):
  # add index back to columns for jsonl
  group_map = df.groupby('object_group').indices
  target_train_frac = 0.6
  target_val_frac = 0.2
  rng = np.random.default_rng(seed)
  # split indices by groups proportional to the size of the group
  train_sizes = tree.map_structure(lambda x: int(target_train_frac * len(x)),
                                   group_map)
  val_sizes = tree.map_structure(lambda x: int(target_val_frac * len(x)),
                                 group_map)

  def split_group(train_size, val_size, indices):
    shuffled = rng.permutation(indices)
    return {
        'train': shuffled[:train_size],
        'validation': shuffled[train_size:train_size + val_size],
        'test': shuffled[train_size + val_size:]
    }

  split_map = tree.map_structure(split_group, train_sizes, val_sizes, group_map)
  index_map = tree.map_structure(lambda *args: np.concatenate(args),
                                 *list(split_map.values()))

  def format_row(row):
    example = {
        'class_id': row['class_id'],
        'display_name': row['display_name'],
        'object_group': row['object_group'],
        'label': row[COLORS].to_dict(),
        'ngram': row['ngram'],
        'vars': row[['object_s', 'areis_s', 'object_pl', 'areis_pl']].to_dict(),
    }
    return example

  data_map = T.valmap(lambda idxs: df.iloc[idxs], index_map)
  data_map = T.valmap(lambda df: df.apply(format_row, axis=1).to_list(),
                      data_map)
  return data_map


def main(_):
  dsd, _ = create_dataset()
  if FLAGS.coda_ds_export_dir is not None:
    for split, ds in dsd.items():
      ds.to_json(
          path_or_buf=os.path.join(
              os.environ.get('BUILD_WORKSPACE_DIRECTORY', ''),
              FLAGS.coda_ds_export_dir,
              f'default_{split}.jsonl',
          ),
          orient='records',
          lines=True,
      )


if __name__ == '__main__':
  app.run(main)
