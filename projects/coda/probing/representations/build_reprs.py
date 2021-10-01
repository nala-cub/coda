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
""" Download a clip or huggingface model """

from typing import Any, List, Optional, Union

from absl import app
from absl import logging
import clip
from datasets import DatasetDict
import torch

import labtools
from probing._src.config_util import hf_auto_configure
from probing._src.configurable import configurable
from probing._src.fwd_steps import feature_extraction_step_single_token
from probing._src.fwd_steps import run_fwd_step
from probing._src.preprocess_steps import apply_preprocessing
from probing._src.preprocess_steps import get_preprocessing_step_for_model
from probing.dataset.dataset import create_dataset


def _run_hf_extraction(ds, tokenizer, model, step_fn, batch_size=64):
  # we want the base model outputting hidden states
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = model.to(device).eval()

  fwd_columns, meta_vars = labtools.hf_get_fwd_columns(ds['train'], model)
  meta_vars += ['label']
  torch_columns = fwd_columns + ['label', 'target_tok_idx']

  ds.set_format('torch', columns=torch_columns, output_all_columns=True)

  # update config to output hidden states.
  model.config.return_dict = True
  model.config.output_hidden_states = True

  maxlen = max([len(eid) for split in ds for eid in ds[split]['input_ids']])
  inner = run_fwd_step(model,
                       tokenizer,
                       step_fn,
                       meta_vars,
                       padding='max_length',
                       max_length=maxlen)
  with torch.no_grad():
    ds = ds.map(inner, batched=True, batch_size=batch_size)
  return ds


def _build_hf_repr_ds(
    model_name: Optional[str] = None,
    batch_size: int = 64,
    use_pretrained: bool = True,
    cache_dir: Optional[str] = None,
    repr_dataset_save_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> DatasetDict:
  """ Build huggingface representations.

  Args:
    model_name: model name or path.
    randinit: use a randomly initialize model instead of a pretrained one.
    model_dir: Model cache directory.
    repr_dataset: directory to save outputs
  """
  ds, _ = create_dataset()
  model_config, tokenizer, model = hf_auto_configure(model_name,
                                                     cache_dir=cache_dir)
  # we have a pretrained one, get rid of it.
  if not use_pretrained:
    logging.warning('Using a randomly initialized configuration of %s',
                    model_name)
    model_cls = model.__class__
    del model
    model = model_cls(model_config)

  if max_examples is not None and max_examples > len(ds):
    logging.warning(
        'received nonnull value for max_examples, filtering the dataset to only'
        'include the first %d examples.', max_examples)
    for split in ds:
      ds[split] = ds[split].select(range(max_examples))

  step_fn = feature_extraction_step_single_token
  example_fn = get_preprocessing_step_for_model(model_config.model_type)

  ds = ds.filter(lambda ex: ex['template_group'] == 1)
  ds = apply_preprocessing(ds, example_fn, tokenizer=tokenizer)
  repr_ds = _run_hf_extraction(ds,
                               tokenizer,
                               model,
                               step_fn,
                               batch_size=batch_size)

  if repr_dataset_save_dir is not None:
    repr_ds.save_to_disk(repr_dataset_save_dir)
  return repr_ds


@torch.no_grad()
def _clip_zeroshot_classifier(
    class_templates: Union[List[str], List[List[str]]],
    model,
):
  """ Creating zero-shot classifier weights from the text encoder.

  Args:
    class_templates: This should be a 1 or 2-d list of strings, where the
      first dimension represents the class. If given, the second dimension
      represents the template variations to average over for that class.
    model: Instance of CLIP.

  Returns:
    zeroshot_weights <num_classes, embedding_size>: The text embedding for each
      class, averaged across templates.
  """
  device = next(model.parameters()).device
  zeroshot_weights = []
  for texts in class_templates:
    texts = clip.tokenize(texts, model.context_length)  # tokenize
    texts = texts.to(device)
    class_embeddings = model.encode_text(texts)  # embed with text encoder
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()
    zeroshot_weights.append(class_embedding)
  zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
  return zeroshot_weights


def _clip_zeroshot_classifier_map(ex, model):
  ex['hidden_states'] = _clip_zeroshot_classifier(ex['text'], model).tolist()
  return ex


def _build_clip_repr_ds(
    model_name: Optional[str] = None,
    use_pretrained: bool = True,
    cache_dir: Optional[str] = None,
    repr_dataset_save_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
    **_: Any,
) -> DatasetDict:
  """ Build clip representations.

  Args:
    model_name: model name or path.
    randinit: use a randomly initialize model instead of a pretrained one.
    cache_dir: Model cache directory.
    repr_dataset: directory to save outputs
    max_examples: max number of examples, used for smoke tests.
  """
  ds, _ = create_dataset()

  if max_examples is not None and max_examples > len(ds):
    logging.warning(
        'received nonnull value for max_examples, filtering the dataset to only'
        'include the first %d examples.', max_examples)
    for split in ds:
      ds[split] = ds[split].select(range(max_examples))
  model, _ = clip.load(model_name,
                       device='cuda',
                       jit=False,
                       download_root=cache_dir)
  # for randinit configs
  if not use_pretrained:
    logging.warning('Using a randomly initialized configuration of %s',
                    model_name)
    model.initialize_parameters()
  ds = ds.filter(lambda ex: ex['template_group'] == 0)
  ds = ds.map(_clip_zeroshot_classifier_map,
              batched=True,
              batch_size=-1,
              fn_kwargs={'model': model})

  if repr_dataset_save_dir:
    ds.save_to_disk(repr_dataset_save_dir)
  return ds


@configurable('build')
def build_repr_ds(
    model_name: Optional[str] = None,
    batch_size: int = 64,
    use_pretrained: bool = True,
    cache_dir: Optional[str] = None,
    force_clip: Optional[bool] = False,
    repr_dataset_save_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> DatasetDict:
  """ Build clip or huggingface reprs.

  Args:
   model_name: model name or path.
    use_pretrained: use a pretrained model instead of a randomly initialized one
    model_dir: Model cache directory.
    repr_dataset: directory to save outputs
    cache_dir: directory to store downloaded models. if not specified will use
      the default location for either library.
    force_clip: by default, we only use load_clip_model if the model is a known
      clip model name. one can use `force_clip=True` to override this behavior.
    max_examples: max number of examples, used for smoke tests.
  """
  if model_name is None:
    raise ValueError('Must provide a model name')
  known_clip_models = ['ViT-B/32', 'RN50', 'RN50x4', 'RN101']
  if force_clip or model_name in known_clip_models:
    logging.info('building clip reprs for %s', model_name)
    builder_fn = _build_clip_repr_ds
  else:
    logging.info('building huggingface reprs for %s', model_name)
    builder_fn = _build_hf_repr_ds
  return builder_fn(
      model_name=model_name,
      use_pretrained=use_pretrained,
      cache_dir=cache_dir,
      max_examples=max_examples,
      repr_dataset_save_dir=repr_dataset_save_dir,
      batch_size=batch_size,
  )


def main(_):
  build_repr_ds()


if __name__ == '__main__':
  app.run(main)
