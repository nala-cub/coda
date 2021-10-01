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
""" Core forward step implementations """
from __future__ import annotations

from typing import Any, Callable, Dict

import cytoolz.curried as T
from einops import rearrange
from einops import reduce
from einops import repeat
import torch
from torch import Tensor
import torch.nn.functional as F
import tree

import labtools

TensorDict = Dict[str, Tensor]
StepFn = Callable[..., TensorDict]


def mlm_step(model, model_input: TensorDict, meta, **_) -> TensorDict:
  """ Predicts the probabilities of a given set of token at a specified index

  Args:
    model: model to evaluate. Should return an object with `logits` as an attr.
    model_input: dictionary contaning all inputs necessary for model to produce
      logit outputs.
    meta: dictionary containing all metadata to be forwarded as output. Must
      contain the following vars for processing:
      `option_input_ids` <batch_size, num_options>: tensor containing the input
        ids to pull from the vocabulary for each example.
      `target_tok_idx` <batch_size>: tensor containing the index to compute
        probabilities for.

  Returns:
    dictionary `output`, with one entry `probs`.
      `probs` <batch_size, num_options>: a tensor containing the probability
        for each token `j` in each example `i` at `probs[i, j]`.
  """
  logits_lm = model(**model_input).logits
  num_options = meta['option_input_ids'].shape[-1]
  # subset by vocab for each sequence in the batch
  option_ids = repeat(meta['option_input_ids'].to(model.device),
                      'b num_options -> b seq_len num_options',
                      seq_len=logits_lm.shape[1])

  logits_subset_options = torch.gather(logits_lm, -1, option_ids)

  # pull the logits for the target token(s)
  # logits_subset_target_tok: <bs, 1, num_options>
  target_tok_idx = repeat(meta['target_tok_idx'].to(model.device),
                          'b -> b m num_options',
                          m=1,
                          num_options=num_options)
  logits_subset_target_tok = torch.gather(logits_subset_options, 1,
                                          target_tok_idx)

  # get probs
  probs_subset_target_tok = F.softmax(logits_subset_target_tok.squeeze(1), -1)
  output = {'probs': probs_subset_target_tok}
  return output


def causal_step_loss(model, model_input: TensorDict, **_) -> TensorDict:
  """ Computes the normalized per-sequence loss with padding adjustments.

  This is different than standard Causal LM loss, where you simply want to
  minimize *all* loss, thus averaging is okay. For probing with loss, we use the
  loss value to select between sequences, and thus need a more accurate value.

  Note that this is only necessary because we are doing this in batches, and
  each example coresponds to an option rather than a sample from the original
  dataset. Thus it is important to average only over the valid tokens for each
  sequence. Otherwise, you could end up with very different results depending
  on which option was in which batch.

  Note that the label values don't matter here and in effect do nothing. we
  don't need the loss to be 0, because they are not considered in the
  aggregation anyway.

  Args:
    model: model to evaluate. Should return an object with `logits` as an attr.
    model_input: dictionary contaning all inputs necessary for model to produce
      logit outputs. Must contain the following vars for processing:
        `attendion_mask` <batch_size, seq_len>: attn mask for the inputs.
          values should be 1 or 0.
        `input_ids` <batch_size, seq_len>`: input ids for the model, also
          shifted to be our labels.

  Returns:
    dictionary `output`, with one entry `loss`.
      `loss` <batch_size>: a tensor containing the loss for each sequence
  """
  # feed through model
  # lm_logits: <bs, seq_len, vocab_size>
  logits_lm = model(**model_input).logits
  # Shift so that tokens < n predict n (from transformers.modeling_gpt2)
  shift_logits = logits_lm[..., :-1, :].contiguous()
  shift_labels = model_input['input_ids'][..., 1:].contiguous()
  # we still want loss to be 0 at the non-valid tokens (padding), bc. we sum
  # set labels at pad indices to -100 s.t. loss=0 for pad
  attn_mask = model_input['attention_mask'][..., 1:]
  shift_labels[attn_mask.eq(0)] = -100
  # Flatten the tokens (no reduction)
  loss = F.cross_entropy(rearrange(shift_logits, 'b s v -> (b s) v'),
                         rearrange(shift_labels, 'b s -> (b s)'),
                         reduction='none',
                         ignore_index=-100)
  # avg loss over each sequence
  # rearrange s.t. we have batches
  loss = rearrange(loss, '(b s) -> b s', b=shift_logits.shape[0])
  # we need the number of 'live' tokens / sequence to get the loss
  nb_tok_per_sq = reduce(attn_mask, 'b s -> b', 'sum')
  loss = loss.sum(-1) / nb_tok_per_sq
  output = {'loss': -loss}
  return output


def feature_extraction_step_single_token(
    model,
    model_input: TensorDict,
    meta,
    **_: Any,
) -> TensorDict:
  """ Extracts the hidden states from the last layer of a model.

  Args:
    model: model to evaluate. Should return and object with `hidden_states` as
      an attr.
    model_input: dictionary contaning all inputs necessary for model to produce
      hidden_states.
    output: dictionary containing all metadata to be forwarded as output.

  Returns:
    Updated dictionary `output`, with a new entry `hidden_states`.
      `hidden_states` <bs,h>: a tensor containing the final hidden
      state for each sequence. for mlm this is the position corr. to the
      mask token, for causal it is the final token (excluding padding).
  """
  model_output = model(**model_input)
  # hidden states from the last layer.
  hidden_states = model_output.hidden_states[-1]
  # extract the token corresponding to the mask, or eos for causal. It is the
  # responsibility of the preprocessing function to provide the location.
  # target_tok_idx: <bs>
  target_tok_idx = meta['target_tok_idx'].to(model.device)
  target_tok_idx = repeat(target_tok_idx,
                          'b -> b m vs',
                          m=1,
                          vs=hidden_states.shape[-1])
  # hidden_states: <bs,1,hidden_size>
  hidden_states = torch.gather(hidden_states, 1, target_tok_idx)
  hidden_states = hidden_states.squeeze(1)
  output = {'hidden_states': hidden_states}
  return output


@T.curry
def run_fwd_step(
    model,
    tokenizer,
    step_fn: StepFn,
    meta_vars: list[str],
    batch: dict[str, Any],
    **kwargs: Any,
) -> dict[str, list]:
  """ Run a forward step

  Args:
    model: model to evaluate. Should return an object with `logits` as an attr.
      this is typically a child of `transformers.PreTrainedModel`, but that is
      not required. The model should have a `device` attribute, which is used
      to transfer the inputs to the same device as the model.
    tokenizer: tokenizer used to pad the sequences, typically a
      `transformers.PreTrainedTokenizer`.
    step_fn: a fwd function, such as `probing._src.mlm_step`. Preprocessing is
      only performed on the variables *not* marked by `meta_vars`, it is
      expected that `step_fn` takes car of processing any other variables it
      needs.
    meta_vars: a list of keys that should be split from the batch and marked as
      meta.
    batch: dictionary contaning all model inputs and meta vars.
    **kwargs: keyword arguments passed to `tokenizer.pad`.

  Returns:
    the output of `step_fn` with all tensors converted into python lists.
  """
  meta, model_input = labtools.split_by_keys(batch, keys=meta_vars)
  model_input = tokenizer.pad(model_input, return_tensors='pt', **kwargs)
  model_input = tree.map_structure(lambda x: x.to(model.device), model_input)
  output = step_fn(model=model, model_input=model_input, meta=meta)
  output = tree.map_structure(lambda x: x.tolist(), output)
  return output
