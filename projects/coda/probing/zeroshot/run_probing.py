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
""" Baseline code for all models. """

from typing import Any, Callable, Dict, Optional

from absl import app
from absl import flags
from absl import logging
import cytoolz.curried as T
import datasets
from datasets import Dataset
from datasets.splits import Split
from pandas import DataFrame
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import labtools
from probing._src.config_util import hf_auto_configure
from probing._src.configurable import configurable
from probing._src.constants import COLORS
from probing._src import fwd_steps
from probing._src.fwd_steps import run_fwd_step
from probing._src.preprocess_steps import apply_preprocessing
from probing._src.preprocess_steps import get_preprocessing_step_for_model
from probing.dataset.dataset import create_dataset

ExampleFn = Callable[[Dict[str, Any], PreTrainedTokenizerBase], Dict[str, Any]]

FLAGS = flags.FLAGS


@T.curry
def _prepare_results(tokenizer: PreTrainedTokenizerBase,
                     example: Dict[str, Any]) -> Dict[str, Any]:
  preds = example.get('probs', None)

  if preds is None:  # GPT 1/2
    example['option'] = tokenizer.decode(example['option_input_ids']).strip()
  else:
    for i, color in enumerate(COLORS):
      example[color] = preds[i]
  example['object_idx'] = example['class_id']
  return example


def _load_coda_dataset(max_examples: Optional[int] = None) -> Dataset:
  ds_d, _ = create_dataset()
  ds = datasets.concatenate_datasets(
      [ds_d[s] for s in (Split.TRAIN, Split.VALIDATION, Split.TEST)])
  ds = ds.filter(lambda ex: ex['template_group'] == 1)  # only text masking
  if max_examples is not None and max_examples > len(ds):
    logging.warning(
        'received nonnull value for max_examples, filtering the dataset to only'
        'include the first %d examples.', max_examples)
    ds = ds.select(range(max_examples))

  return ds


@configurable
def run_zeroshot(
    model_name: Optional[str] = None,
    batch_size: int = 64,
    pred_fpath: Optional[str] = None,
    model_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> DataFrame:
  """ Run zeroshot probing.

  Args:
    model_name: name or path of the model
    batch_size: batch size to use for probing
    model_dir: Model cache directory
    preds_csv: File to write predictions to.
    max_examples: maximum number of examples, only used for smoke tests.
  """
  model_config, tokenizer, model = hf_auto_configure(model_name,
                                                     cache_dir=model_dir)

  FWD_STEP_FNS = {
      'albert': fwd_steps.mlm_step,
      'roberta': fwd_steps.mlm_step,
      'gpt2': fwd_steps.causal_step_loss,
  }
  step_fn = FWD_STEP_FNS.get(model_config.model_type, None)

  # adjust for the gpt2 types
  example_sname = model_config.model_type
  if example_sname == 'gpt2':
    example_sname += '-loss'

  example_fn = get_preprocessing_step_for_model(example_sname)

  if step_fn is None:
    logging.warning('Unknown model type %s', model_config.model_type)
  batched_preprocessing = model_config.model_type == 'gpt2'

  ds = _load_coda_dataset(max_examples)
  ds = apply_preprocessing(ds,
                           example_fn,
                           batched=batched_preprocessing,
                           tokenizer=tokenizer)  ## had params

  # for selecting the correct elements of `batch`
  fwd_columns, meta_vars = labtools.hf_get_fwd_columns(ds, model)
  torch_meta_vars = ['target_tok_idx', 'template_idx', 'option_input_ids']

  torch_columns = fwd_columns + torch_meta_vars

  ds.set_format('torch', columns=torch_columns, output_all_columns=True)

  meta_vars += ['label']

  model.config.return_dict = True

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # run
  model = model.to(device)
  model = model.eval()

  inner = run_fwd_step(model, tokenizer, step_fn, meta_vars, padding='longest')

  with torch.no_grad():
    ds = ds.map(inner, batched=True, batch_size=batch_size)
    ds = ds.map(_prepare_results(tokenizer))

  # gpt 1/2 have 1 example per option. other models have 1 example per example
  # these will have `color` and `loss`, which need to be pivotted.
  if model_config.model_type in ('openai-gpt', 'gpt2'):
    keys = ['class_id', 'template_idx', 'option', 'loss']
    ds.set_format('pd', columns=keys)
    preds = ds[:]
    # pivot
    preds = preds.pivot(index=['class_id', 'template_idx'],
                        columns='option',
                        values='loss').reset_index()
  else:
    # ic(shapes)
    keys = ['class_id', 'template_idx', *COLORS]
    ds.set_format('pd', columns=keys)
    preds = ds[:]

  preds['name'] = model_name
  if pred_fpath is not None:
    preds.to_csv(pred_fpath, index=False)

  return preds


def main(_):
  run_zeroshot()


if __name__ == '__main__':
  app.run(main)
