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
""" Provides utilities for working with `datasets` and `transformers`"""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any, Callable, Optional

from absl import logging


def hf_one_to_many(fn: Callable) -> Callable:
  """ Wrappes a huggingface map for 1 -> many ops."""

  @wraps(fn)
  def _one_to_many(examples, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
    records = []
    num_examples = len(examples[list(examples)[0]])
    for i in range(num_examples):
      out = fn({k: v[i] for k, v in examples.items()}, *args, **kwargs)
      records.extend(out)

    return {k: [r[k] for r in records] for k in records[0].keys()}

  return _one_to_many


def hf_get_fwd_columns(ds, model, description: Optional[str] = None):
  """ Split off meta columns (not for model.forward)

  Adapted with modifications from `transformers.trainer._remove_unused_columns`.

  Args:
    ds (datasets.Dataset): HF Dataset
    model (transformers.PreTrainedModel): HF Model
  """
  # Inspect model forward signature to keep only the arguments it accepts.
  signature = inspect.signature(model.forward)
  signature_columns = list(signature.parameters.keys())
  # Labels may be named label or label_ids, the default data collator handles
  # that.
  signature_columns += ['label', 'label_ids']
  fwd_columns = [k for k in signature_columns if k in ds.column_names]
  meta_columns = list(set(ds.column_names) - set(signature_columns))
  if len(meta_columns) > 0:
    dset_desc = '' if description is None else f'in the {description} set '
    logging.info(
        'The following columns %s don\'t have a corresponding '
        'argument in `%s.forward` and have been marked as meta: %s', dset_desc,
        model.__class__.__name__, ', '.join(meta_columns))
  return fwd_columns, meta_columns
