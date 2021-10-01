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
""" Tests for the core forward step implementations. """
from __future__ import annotations

from dataclasses import dataclass
import zlib

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from einops import repeat
import numpy as np
import torch
from torch import Tensor
import tree

from probing._src.fwd_steps import causal_step_loss
from probing._src.fwd_steps import mlm_step

FLAGS = flags.FLAGS


@dataclass
class DummyModelOutput:
  logits: Tensor


@dataclass
class DummyModel:
  logits: Tensor

  def __post_init__(self):
    self.device = self.logits.device

  def __call__(self, **kwargs):
    return DummyModelOutput(self.logits)


class TestMlmStep(parameterized.TestCase):
  """ Test mlm_step """

  def test_oid_order(self):
    """ Order of option ids shouldn't matter. """
    bs = 2
    vs = 128
    seq_len = 40

    option_iids = torch.tensor([[0, 1, 2, 3], [3, 2, 0, 1]])
    logits = torch.full((bs, seq_len, vs), -np.inf)
    # set only the lo

    target_tok_idx = torch.tensor([10, 20])

    logits[0, 10, 1] = 1
    logits[1, 20, 1] = 1

    # setup inputs
    model = DummyModel(logits=logits)
    model_input = {}
    output = {
        'option_input_ids': option_iids,
        'target_tok_idx': target_tok_idx,
    }

    res = mlm_step(model, model_input, output)

    expected_probs = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    pred_probs = res['probs'].numpy()

    np.testing.assert_almost_equal(pred_probs, expected_probs)


class TestCausalStepLoss(parameterized.TestCase):
  """ Test causal_step_loss """

  def setUp(self):
    """ setup rng for random tests """
    super().setUp()
    # See: https://github.com/google/jax/blob/main/jax/test_util.py
    self.rng = np.random.RandomState(zlib.adler32(
        self._testMethodName.encode()))

  def test_manual_example(self):
    """ manual test for padding """

    bs = 3
    vs = 2
    seq_len = 9

    # 3 sequences where 1 full, 2 is half, and 3 is empty.
    attn_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
    ])

    # set as all zeros
    # shape: <bs seq_len vocav>
    logits_lm = torch.zeros((bs, seq_len, vs))

    # always predict the wrong value this will be the value of the loss on
    # non-pad tokens. we choose numbers here to make the math easier.
    logits_lm[..., 1] = 100
    input_ids = torch.zeros((bs, seq_len), dtype=torch.long)

    model = DummyModel(logits=logits_lm)

    model_input = {
        'input_ids': input_ids,
        'attention_mask': attn_mask,
    }
    output = {}

    output = causal_step_loss(model, model_input, meta=output)
    loss = output['loss'].cpu().numpy()
    np.testing.assert_equal(loss, -100)

  def test_nonbatched_equivalence(self):
    """ tests that the per-example loss is the same is w/o padding """
    rng = self.rng

    bs = 100
    vs = 200
    max_seq_len = 100

    seq_lens = rng.randint(2, max_seq_len - 1, bs)

    # 3 sequences where 1 full, 2 is half, and 3 is empty.
    attn_mask = np.ones((bs, max_seq_len))
    for i, s in enumerate(seq_lens):
      attn_mask[i, s:] = 0

    # create random inputs and outputs
    logits_lm = rng.rand(bs, max_seq_len, vs)
    input_ids = rng.randint(0, vs - 1, (bs, max_seq_len), dtype=np.int64)
    model = DummyModel(logits=torch.from_numpy(logits_lm))

    model_input = {
        'input_ids': torch.from_numpy(input_ids),
        'attention_mask': torch.from_numpy(attn_mask),
    }
    output = {}

    output = causal_step_loss(model, model_input, meta=output)
    batched_loss = output['loss'].cpu().numpy()

    # now manually
    nobatched_loss = []
    for i, s in enumerate(seq_lens):
      attn_mask[i, s:] = 0
      model_input = {
          'input_ids': input_ids[i:i + 1, :s],
          'attention_mask': attn_mask[i:i + 1, :s],
      }
      model = DummyModel(logits=torch.from_numpy(logits_lm[i:i + 1, :s, :]))
      # ->
      model_input = tree.map_structure(torch.from_numpy, model_input)
      output = causal_step_loss(model, model_input, meta={})
      # <-
      nobatched_loss.append(output['loss'][0].cpu().numpy())

    np.testing.assert_almost_equal(batched_loss, nobatched_loss)


class TestFeatureExtractionStep(absltest.TestCase):
  """ Test for huggingface feature extraction step """

  def test_extracts_correct_indices(self):
    bs = 6
    seq_len = 10
    vs = 8

    target_tok_idx = torch.tensor([1, 9, 2, 3, 4, 5])

    # make a grid tensor
    x = np.ones((bs, seq_len, vs))

    # batch - diff by 100
    bs_mult = 100 * np.arange(bs).reshape(-1, 1, 1)
    x *= bs_mult

    # token - diff by 10
    x += 10 * np.arange(seq_len).reshape(-1, 1)

    # vocab - diff by 1
    x += 1 * np.arange(vs).reshape(-1)

    x = torch.from_numpy(x)

    expected = np.asarray([[[10., 11., 12., 13., 14., 15., 16., 17.]],
                           [[190., 191., 192., 193., 194., 195., 196., 197.]],
                           [[220., 221., 222., 223., 224., 225., 226., 227.]],
                           [[330., 331., 332., 333., 334., 335., 336., 337.]],
                           [[440., 441., 442., 443., 444., 445., 446., 447.]],
                           [[550., 551., 552., 553., 554., 555., 556., 557.]]],)

    # x
    target_tok_idx = repeat(target_tok_idx, 'b -> b m vs', m=1, vs=vs)
    sliced = torch.gather(x, 1, target_tok_idx)

    np.testing.assert_almost_equal(sliced, expected)


if __name__ == '__main__':
  absltest.main()
