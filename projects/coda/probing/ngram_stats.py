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
""" Ngram statistics """
from typing import Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import pandas as pd

from probing._src.configurable import configurable
from probing._src.constants import COLORS
from probing._src.metrics import compute_metrics
from probing.dataset.dataset import create_dataset


@configurable('ngrams')
def ngram_stats(
    ngram_gbc_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    ngram_metrics_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """ Collect Statistics for the Google Books Ngram Corpus

  Args:
    ngram_gbc_path: Filtered Ngram counts from the GBC.
    metadata_path: File containing dataset metadata.
    ngram_metrics_path: File to write ngram metrics to.

  Returns:
    gbc_ngram_metrics: ngram metcis for GBC
  """

  if metadata_path is None:
    logging.info('did not receive metadata, regenerating the dataset.')
    _, meta = create_dataset()
  else:
    meta = pd.read_csv(metadata_path)

  # load per-object ngram counts from the google books corpus
  gbc_ngram_counts = pd.read_csv(ngram_gbc_path)

  # now compute per object stats.
  # this seems to alter the ngram_counts df..
  gbc_ngram_metrics = compute_per_object_metrics(gbc_ngram_counts, meta)

  # save for use by other steps
  if ngram_metrics_path:
    gbc_ngram_metrics.to_csv(ngram_metrics_path, index=False)
  return gbc_ngram_metrics


def compute_per_object_metrics(ngrams, meta):
  # it is essential that ngrams and meta are aligned before we compute the
  # metrics otherwise this is rubbish.
  ngrams = ngrams.set_index('class_id', verify_integrity=True).sort_index()
  meta = meta.set_index('class_id', verify_integrity=True).sort_index()
  # ngrams may be superset of the actual class_ids we care about, that's okay.
  ngrams = ngrams.loc[meta.index]
  # but now they must be the same.
  assert (
      ngrams.index == meta.index).all(), 'ngrams does not contain all values.'

  # convert ngram counts -> probs
  ngrams[COLORS] = ngrams[COLORS].div(ngrams[COLORS].sum(1), axis=0)
  ngrams[COLORS] = ngrams[COLORS].fillna(0)

  preds = ngrams[COLORS].values
  references = meta[COLORS].values

  metrics = list(map(compute_metrics, preds, references))
  # make metrics df w/ metadata. note this only works bc. of our alignment.
  dfm = pd.DataFrame(metrics)
  dfm.index = meta.index
  dfm['object_group'] = meta['object_group']
  dfm = dfm.reset_index()

  return dfm


def main(_):
  logging.get_absl_handler().use_absl_log_file('ngram_stats')
  ngram_stats()


if __name__ == '__main__':
  flags.mark_flags_as_required(['ngrams.ngram_gbc_path'])
  app.run(main)
