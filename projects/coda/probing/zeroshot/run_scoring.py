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
""" Contains the scoring for zeroshot probing """

from typing import Optional

from absl import app
from absl import flags
import pandas as pd
from pandas import DataFrame

import labtools
from probing._src.configurable import configurable
from probing._src.constants import COLORS
from probing._src.metrics import compute_metrics

try:
  from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
  ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def compute_per_object_metrics(preds, meta):
  ic(meta.columns, meta)
  df = preds.merge(meta, on='class_id', suffixes=('_pred', ''))

  pred_cols = [f'{c}_pred' for c in COLORS]
  ref_cols = COLORS

  preds = df[pred_cols].values
  references = df[ref_cols].values

  metrics = labtools.safe_map(compute_metrics, preds, references)
  # make metrics df w/ metadata
  dfm = pd.DataFrame(metrics)
  dfm = dfm.join(df[['object_group', 'class_id', 'template_idx']])

  # upper-bound: take the 'best' results for a given object, dict. by KT
  dfm = dfm.sort_values('kendalls_tau', ascending=False)
  # .head(1) takes the first row of each group, maintaining the og sorted index
  dfm = dfm.groupby(['class_id']).head(1)
  return dfm


def zeroshot_scoring(
    preds: DataFrame,
    meta: DataFrame,
    ngrams: DataFrame,
    savepath: Optional[str] = None,
) -> DataFrame:
  df = compute_per_object_metrics(preds=preds, meta=meta)

  # convert ngram counts -> probs
  ngrams[COLORS] = ngrams[COLORS].div(ngrams[COLORS].sum(1), axis=0)
  # if the values are NaNs, this means the ngram was never present.
  ngrams[COLORS] = ngrams[COLORS].fillna(0)

  # add object group to ngrams
  ngrams = ngrams.merge(meta[['class_id', 'object_group']],
                        on='class_id',
                        how='left',
                        validate='one_to_one')

  df_ngrams = compute_per_object_metrics(preds=preds, meta=ngrams)
  # align the two dfs
  df = df.set_index('class_id', verify_integrity=True).sort_index()
  df_ngrams = df_ngrams.set_index('class_id',
                                  verify_integrity=True).sort_index()
  # take the avg_corr, and save as 'ngram_label_corr_avg'. Note that `ngram`
  # cannot be a suffix here.
  df['ngram_label_corr_avg'] = df_ngrams['corr_avg']
  if savepath is not None:
    df.to_csv(savepath)
  return df


@configurable('zeroshot_scoring')
def zeroshot_scoring_cmd(
    preds_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    ngram_gbc_path: Optional[str] = None,
    savepath: Optional[str] = None,
) -> DataFrame:
  """ Run zeroshot scoring

  Args:
    metadata_path: File containing dataset metadata.
    preds_path: File containing zeroshot preds for 1 model.
    ngram_gbc_path: Ngram counts from the google books corpus.
    savepath: File to save results in, if provided.
  """
  # compute the real metrics
  preds = pd.read_csv(preds_path)
  meta = pd.read_csv(metadata_path)
  ngrams = pd.read_csv(ngram_gbc_path)

  return zeroshot_scoring(preds, meta, ngrams, savepath)


def main(_):
  zeroshot_scoring_cmd()


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'zeroshot_scoring.metadata_path',
      'zeroshot_scoring.preds_path',
      'zeroshot_scoring.ngram_gbc_path',
  ])
  app.run(main)
