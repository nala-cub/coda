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
""" Preprocessing steps for Huggingface Models. """
from __future__ import annotations

import copy
import os
import re
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from absl import flags
from absl import logging
from datasets import Dataset
from datasets import DatasetDict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from probing._src.constants import COLORS

Tok = TypeVar('Tok')
_DS = TypeVar('_DS', Dataset, DatasetDict)
Example = Dict[str, str]
ExampleBatch = Dict[str, List[str]]
Tokenizer = PreTrainedTokenizerBase
PreprocessFn = Callable[[Example, Tokenizer], Example]

BatchedPreprocessFn = Callable[[ExampleBatch, Tokenizer], List[ExampleBatch]]

FLAGS = flags.FLAGS

flags.DEFINE_boolean('ds_cache_files', True, 'Use Dataset cache files.')

__all__ = (
    'apply_preprocessing',
    'get_option_encodings',
    'create_example_gpt',
    'create_example_roberta',
    'create_example_albert',
    'create_examples_gpt_loss',
)


def get_preprocessing_step_for_model(
    step_name: str) -> Union[PreprocessFn, BatchedPreprocessFn]:
  example_gen_fns = {
      'albert': create_example_albert,
      'roberta': create_example_roberta,
      'gpt2': create_example_gpt,
      'gpt2-loss': create_examples_gpt_loss,
  }

  example_fn = example_gen_fns.get(step_name, None)
  if example_fn is None:
    raise ValueError('Unknown step name %s' % step_name)
  return example_fn


def apply_preprocessing(
    ds: _DS,
    p_example_fn: Union[PreprocessFn, BatchedPreprocessFn],
    batched: bool = False,
    batch_size: int = 1,
    use_cache_files: bool = False,
    **fn_kwargs: Any,
) -> _DS:
  """Apply preprocessing to a dataset

  Args:
    ds: dataset to apply preprocessing to
    p_example_fn: Example map function. this should be a `PreprocessFn` if
      `batched` is `False`, else `BatchedPreprocessFn`.
    batch_size: batch_size for processing the data
    use_cache_files: Predicate indicating the sue of cache files.
    """

  logging.info('Using preprocessing fn: %s', p_example_fn.__name__)

  # create examples
  prev_parallelism = os.environ.get('TOKENIZERS_PARALLELISM', 'true')
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  ds = ds.map(
      p_example_fn,
      batch_size=batch_size,
      batched=batched,
      with_indices=False,
      fn_kwargs=fn_kwargs,
      load_from_cache_file=use_cache_files,
  )
  os.environ['TOKENIZERS_PARALLELISM'] = prev_parallelism
  return ds


def try_get_index(tokens: list[Tok],
                  stoken: Tok,
                  s: int = 0,
                  e: Optional[int] = None,
                  level: int = -2) -> int:
  """ Locates the index of a token in a (sub) sequence of tokens.

  Returns:
    Index of `stoken` in `tokens` if `stoken` is present in `tokens. Otherwise
    returns -1.
  """
  try:
    target_tok_idx = tokens.index(stoken, s, len(tokens) if e is None else e)
    return target_tok_idx
  except ValueError:
    logging.log(level, 'Could not locate token (%s). Tokens: \n%s', stoken,
                tokens)
    return -1


def get_option_encodings(option_words,
                         tokenizer: Tokenizer,
                         strict: bool = False,
                         allow=None):
  fmt_str = ' %s'

  option_encodings = [
      tokenizer(fmt_str % tok, add_special_tokens=False) for tok in option_words
  ]
  if strict:
    _check_encodings(option_encodings, allow=allow, option_words=option_words)
  return option_encodings


def _check_encodings(
    encodings,
    level: Union[int, str] = 'FATAL',
    **kwargs: str,
) -> None:
  # check option_ids
  oid_lens = [len(o.input_ids) for o in encodings]

  if not all(map(lambda x: x == 1, oid_lens)):
    # create table
    err_mapping = {
        str(o.tokens()): o.input_ids for o in encodings if len(o.input_ids) > 1
    }
    info_str = '&'.join(['='.join([k, v]) for k, v in kwargs.items()])
    logging.log(  # pylint: disable=logging-not-lazy
        level,
        '[%s] Found invalid options, expected each to have exatly one token. ' +
        'Encodings: %s', info_str, err_mapping)


def create_example_gpt(example: Example, tokenizer: Tokenizer) -> Example:
  """ Create example for GPT-[1,2]

    Substitute mask with <pad>, and store where the token was placed. During
    repr. collection, we'll need it.

    Example:
      >>> x = 'Most apples are [MASK].'
      ... tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
      ... ex = create_example_gpt({'text': x}, tokenizer)
      ... ex
          {
            'tokens': [
              '|endoftext|',
              '▁most',
              '▁apples',
              '▁are',
              '|endoftext|',
              '|endoftext|',
            ]
            'target_tok_idx': 4,
          }

  """
  # substitute mask with |endoftext| and get rid of all subsequent tokens.
  # during inference, we are predicting the masked token and ignoring everything
  # else anyway for gpt2. Note bos=eos=unk
  text_input = re.sub(r'(( \[MASK\])|(\[MASK\])).+', tokenizer.pad_token,
                      example['text'])
  # fix the text input, which is missing <bos> and <eos> tags.
  # note that these aren't considered "special tokens" by the tokenizer, so we
  # need to adjust the word index.
  if tokenizer.name_or_path.startswith('gpt'):
    text_input = f'{tokenizer.bos_token}{text_input}{tokenizer.eos_token}'

  # get BatchEncoding
  instance = tokenizer(text=text_input,
                       padding='do_not_pad',
                       add_special_tokens=True)

  # locate position of our mask token. note this is actually the same token as
  # eos, so it will always be the last token. this is trivial for a single
  # sequence but makes it easier once we do batch predictions.
  target_tok_idx = try_get_index(instance.tokens(), tokenizer.pad_token, -2,
                                 None)
  # since we use the target_tok_idx to extract hidden states at a given pos.
  # we keep track of the prior index (the one predicting the token of interest).
  target_tok_idx -= 1
  instance['target_tok_idx'] = target_tok_idx
  return instance


def create_examples_gpt_loss(
    examples: ExampleBatch,
    tokenizer: Tokenizer,
) -> ExampleBatch:
  """ Create example for GPT-[1,2]

    This is batchd so that we can expand each each example in the base dataset
      to N examples, where N=number of options.
  """
  example = {k: vals[0] for k, vals in examples.items()}
  # substitute mask with <pad>
  text_input = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.pad_token,
                      example['text'])
  #
  # fix the text input, which is missing <bos> and <eos> tags.
  if tokenizer.name_or_path.startswith('gpt'):
    text_input = f'{tokenizer.bos_token}{text_input}{tokenizer.eos_token}'

  # get BatchEncoding
  instance = tokenizer(text=text_input,
                       padding='do_not_pad',
                       add_special_tokens=True)
  #
  # locate pad token
  target_tok_idx = try_get_index(instance.tokens(), tokenizer.pad_token, 1, -1)
  # add example-general info
  instance['target_tok_idx'] = target_tok_idx

  # hack to resolve 1-to-many issues
  for k, v in example.items():
    if k not in instance:
      instance[k] = v

  # get option ids
  option_encodings = get_option_encodings(COLORS, tokenizer, True)
  option_input_ids = [o.input_ids[0] for o in option_encodings]

  # substitute <pad> with each option
  instances = []
  for i, option_input_id in enumerate(option_input_ids):
    option_instance = copy.deepcopy(instance)
    # index of this option (A -> 0, B-> 1, etc.)
    # option_instance['option_idx'] = i
    # update ids
    option_instance['input_ids'][target_tok_idx] = option_input_id
    option_instance['option_input_ids'] = option_input_id
    # label for this option
    option_instance['label'] = example['label'][i]

    instances.append(option_instance)

  return {k: [r[k] for r in instances] for k in instances[0].keys()}


def create_example_roberta(
    example: Example,
    tokenizer: Tokenizer,
) -> Example:
  """ RoBERTa expects <mask> to be contain the space, e.g. `<mask>=' hi'`. """

  text_input = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.mask_token,
                      example['text'])

  # get encoding
  instance = tokenizer(text=text_input,
                       padding='do_not_pad',
                       add_special_tokens=True)
  # locate pad token
  target_tok_idx = try_get_index(instance.tokens(), tokenizer.mask_token)

  # add example-general info
  instance['target_tok_idx'] = target_tok_idx
  # get color input_ids
  option_encodings = get_option_encodings(COLORS, tokenizer, True)
  option_input_ids = [o.input_ids[0] for o in option_encodings]
  instance['option_input_ids'] = option_input_ids

  return instance


def create_example_albert(example: Example, tokenizer: Tokenizer) -> Example:
  """Create examples for Albert.
    Albert uses whole-word masking, so [MASK] should be replaced with the
    number of tokens that the option has. This version only accounts for SINGLE
    token masking, see create_examples_albert_wwm for details on the other
    approach.

  """
  # substitute [MASK]
  text_input = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.mask_token,
                      example['text'])

  # get BatchEncoding
  instance = tokenizer(text=text_input,
                       padding='do_not_pad',
                       add_special_tokens=True)
  # locate pad token
  target_tok_idx = try_get_index(instance.tokens(), tokenizer.mask_token)

  # add example-general info
  instance['target_tok_idx'] = target_tok_idx
  # get color input_ids
  option_encodings = get_option_encodings(COLORS, tokenizer, True)
  option_input_ids = [o.input_ids[0] for o in option_encodings]
  instance['option_input_ids'] = option_input_ids

  return instance
