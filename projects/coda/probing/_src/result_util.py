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
""" Utility functions forprocessing results """
from __future__ import annotations

from typing import Optional

from absl import logging
import numpy as np
import pandas as pd

from probing._src.constants import DF_PRETTY_COL_NAMES
from probing._src.constants import DF_REPLACE_MAP
from probing._src.constants import METRIC_PRECISION_MAP
from probing._src.constants import MODEL_ORDER
from probing._src.constants import NO_STD_METRICS


def pretty_fmt(  # pylint: disable=dangerous-default-value
    df: pd.DataFrame,
    column_map: Optional[dict[str, str]] = DF_PRETTY_COL_NAMES,
):
  df.columns.name = ''
  df = df.replace(DF_REPLACE_MAP)
  if column_map:
    df = df.rename(columns=column_map)
    if isinstance(df.index, pd.MultiIndex):
      df.index.names = [column_map.get(n, n) for n in df.index.names]
    else:
      df.index.name = column_map.get(df.index.name, df.index.name)

  return df


def scale_metrics(df: pd.DataFrame) -> pd.DataFrame:
  for m in [
      'spearman', 'pearson', 'kendalls_tau', 'acc', 'corr_avg',
      'ngram_label_corr_avg'
  ]:
    if m in df.columns:
      df[m] = df[m] * 100
  return df


def get_order(x: list[str]) -> list[int]:
  return [MODEL_ORDER.index(xi) for xi in x]


def _fmt_mean_with_std_str(x, skip_std: bool = False, precision: str = '.2f'):
  """ Attempts to format a row with +- std.  and add bolding

  All values are rounded to 1 decimal place.

  Formats:
    If `x` only contains the mean, or mean and `x['best'] = False`:
    .. math::
      $\\bar x$
    If `x` only contains the mean and best with `x['best'] = True`:
    .. math::
      $\\textbf{\\bar x}$
    If `x` contains mean and std or mean, std, and `x['best'] = False`:
    .. math::
      $ \\bar x \\pm \\text{std}(x) $
    If `x` contains mean, std, and `x['best'] = True`
    .. math::
      $ \\textbf{\\bar x} \\pm \\textbf{\\text{std}(x)} $

  Args:
    x: row containing at least the `mean` value.
    skip_std: predicate indicating whether the std. value should be ignored.

  """
  mean = x['mean']
  std = x.get('std', np.nan)
  if not np.isfinite(mean):
    return '--'
  mean_str = f'{mean:0.{precision}f}'
  # no standard dev.
  if skip_std or np.isnan(std):
    # + best
    if x.get('best', False):
      return '$\\textbf{%s}$' % mean_str
      # return '$\\textbf{%s} \\quad $' % mean_str
    # - best
    else:
      return '$%s$' % mean_str
      # return '$%s \\quad $' % mean_str
  std_str = f'{std:0.{precision}f}'
  if x.get('best', False):
    return '\\boldpm{%s}{%s}' % (mean_str, std_str)
  return '\\flatpm{%s}{%s}' % (mean_str, std_str)


def fmt_mean_with_std(row):
  cols = [i[0] for i in row.index]
  res = pd.Series({
      metric: _fmt_mean_with_std_str(
          row[metric],
          metric in NO_STD_METRICS,
          METRIC_PRECISION_MAP.get(metric, 2),
      ) for metric in cols
  })
  return res


def get_should_bold(df: pd.DataFrame, groupby=None) -> pd.DataFrame:
  if groupby is not None:
    dfg = df.groupby(groupby)
    cols = [i[0] for i in df.columns if 'mean' in i]
    for col in cols:
      df[(col, 'best')] = False
      if col in (
          'kl_div',
          '$\\mathbf{D_{KL}} \\downarrow$',
          'jensenshannon_div',
      ):
        boldrow = dfg.idxmin()[(col, 'mean')]
      else:
        boldrow = dfg.idxmax()[(col, 'mean')]
      # don't bold if there are NaNs present
      if not boldrow.isna().any():
        df.at[boldrow, (col, 'best')] = True
  else:
    cols = [i[0] for i in df.columns]
    for col in cols:
      df[(col, 'best')] = False
      if len(df[(col, 'mean')]) == 0:
        logging.warn('Detected empty mean for row %s, skipping best', col)
        continue
      elif col in ('kl_div', 'jensenshannon_div'):
        boldrow = df[(col, 'mean')].idxmin()
      else:
        boldrow = df[(col, 'mean')].idxmax()
      df.at[boldrow, (col, 'best')] = True
  return df


def add_ngram_delta_corrs(df: pd.DataFrame,
                          ngram_metrics: pd.DataFrame) -> pd.DataFrame:
  """ Adds ngram delta correlation metrics to a dataframe. """
  df = df.merge(ngram_metrics,
                on=['class_id', 'object_group'],
                suffixes=['', '_ngram'])
  # get metric names
  metric_cols = {
      c: f'{c}_ngram' for c in df.columns if f'{c}_ngram' in df.columns
  }

  # get delta for each metric
  for model_k, ngram_k in metric_cols.items():
    df[f'delta_{model_k}'] = (df[model_k] - df[ngram_k]) * 100

  df = df.drop(columns=metric_cols.values())
  return df


def get_best_model_names(
    df: pd.DataFrame,
    metric: str = 'kendalls_tau',
    higher_is_better: bool = True,
) -> list[str]:
  """ Get the best models per model type, based on some metric"""
  # get the mean and std. for each metric, first over objects of the same group
  df = df.groupby(['model_name', 'model_type', 'model_size',
                   'object_group']).agg(['mean', 'std'])
  # then across groups.
  df = df.groupby(['model_name', 'model_type', 'model_size']).mean()
  # then take the best of each model type
  df = df.sort_values([(metric, 'mean')], ascending=not higher_is_better)
  df = df.reset_index().groupby(['model_type']).head(1)

  best_model_names = df['model_name'].unique()
  return best_model_names
