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
""" Contains the core evaluation metrics  """

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from scipy import stats


def compute_metrics(preds, targets):
  """ Compute all metrics for a given example

  Args:
    preds: array containing predicted probabilities for each class
    targets:  array containing groundtruth probabilities for each class
  """

  spearman, spearman_p = stats.spearmanr(preds, targets)
  kendalls_tau, kendalls_tau_p = stats.kendalltau(preds, targets)
  metrics = {
      'spearman': spearman,
      'spearman_p': spearman_p,
      'kendalls_tau': kendalls_tau,
      'kendalls_tau_p': kendalls_tau_p,
      'corr_avg': (spearman + kendalls_tau) / 2,
      'jensenshannon_dist': jensenshannon(preds, targets),
      'jensenshannon_div': jensenshannon_div(preds, targets),
  }
  metrics['acc'] = int(np.argmax(preds) == np.argmax(targets))
  return metrics


def jensenshannon_div(p, q, base=None, *, axis=0, keepdims=False):
  """ Jensen Shannon Divergence.

  Based on scipy.distance.jensenshannon, without the sqrt.
  """
  p = np.asarray(p)
  q = np.asarray(q)
  p = p / np.sum(p, axis=axis, keepdims=True)
  q = q / np.sum(q, axis=axis, keepdims=True)
  m = (p + q) / 2.0
  left = rel_entr(p, m)
  right = rel_entr(q, m)
  left_sum = np.sum(left, axis=axis, keepdims=keepdims)
  right_sum = np.sum(right, axis=axis, keepdims=keepdims)
  js = left_sum + right_sum
  if base is not None:
    js /= np.log(base)
  return js / 2.0
