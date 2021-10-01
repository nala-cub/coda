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
""" Test Preprocessing Steps for Huggingface Models. """

from absl.testing import absltest
from absl.testing import parameterized
from transformers import AutoTokenizer

from probing._src.preprocess_steps import create_example_albert
from probing._src.preprocess_steps import create_example_gpt
from probing._src.preprocess_steps import create_example_roberta
from probing._src.preprocess_steps import create_examples_gpt_loss
from probing._src.preprocess_steps import get_option_encodings
from probing._src.preprocess_steps import get_preprocessing_step_for_model
from probing._src.preprocess_steps import try_get_index

_COLORS = [
    'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple',
    'red', 'white', 'yellow'
]

_MODEL_NAMES = [
    'gpt2',
    'roberta-base',
    'albert-base-v1',
    'albert-base-v2',
    'gpt2-medium',
    'gpt2-large',
    'roberta-large',
    'albert-large-v1',
    'albert-large-v2',
    'gpt2-xl',
    'albert-xlarge-v1',
    'albert-xlarge-v2',
    'albert-xxlarge-v1',
    'albert-xxlarge-v2',
]


def get_tokenizers():
  return [(name, lambda name=name: AutoTokenizer.from_pretrained(name))
          for name in _MODEL_NAMES]


class TestTryGetIndex(parameterized.TestCase):
  """ test for the index locator function """

  @parameterized.named_parameters(
      ('first_token_single', [0, 5, 2, 1, 3], 0, 0, None, 0),
      ('first_token_multi', [0, 2, 3, 0, -1], 0, 0, None, 0),
      ('first_token_all', [0, 0, 0, 0, 0], 0, 0, None, 0),
      ('last_token', [0, 5, 2, 1, 3], 3, 0, None, 4),
  )
  def test_try_get_index_valid(self, tokens, stoken, s, e, expected_idx):
    """ Test try_get_index where e > stoken position"""
    res_index = try_get_index(tokens, stoken, s, e)
    self.assertEqual(res_index, expected_idx)

  @parameterized.named_parameters(
      ('first_token_single', [0, 5, 2, 1, 3], 1, 0, 3, -1),
      ('first_token_multi', [0, 2, 3, 0, -1], 2, 0, 1, -1),
      ('first_token_all', [0, 0, 0, 0, 0], 0, 0, 0, -1),
      ('last_token', [0, 5, 2, 1, 3], 2, 0, 2, -1),
  )
  def test_try_get_index_error(self, tokens, stoken, s, e, expected_idx):
    """ Test try_get_index e < stoken position"""
    res_index = try_get_index(tokens, stoken, s, e)
    self.assertEqual(res_index, expected_idx)


class TestOptionEncodings(parameterized.TestCase):
  """ test for option encodings """

  @parameterized.named_parameters(*get_tokenizers())
  def test_single_token_options(self, tokenizer):
    """ check that options are encoded properly as single tokens for each model

    It is necessary to check *every* model vocabulary, because it's certainly
    possible words are present in some vocabulary but not others. This is
    generally less important for testing the example processing, but is worth
    checking for the options.
    """
    tokenizer = tokenizer()
    encodings = get_option_encodings(_COLORS, tokenizer, True)
    # check for lost entries that would go undetected by zip.
    self.assertEqual(len(encodings), len(_COLORS))
    for c, encoding in zip(_COLORS, encodings):
      self.assertEqual(
          len(encoding.input_ids),
          1,
          msg='option encoding error for %s' % c,
      )

    # check that options are encoded the same w/ a period.
    p_colors = ['%s.' % c for c in _COLORS]
    p_encodings = get_option_encodings(p_colors, tokenizer, False)
    # check for lost entries that would go undetected by zip.
    self.assertEqual(len(p_encodings), len(_COLORS))
    # now we want 2 tokens, 1 for the period and 1 for the color.
    for c, encoding in zip(_COLORS, p_encodings):
      self.assertEqual(
          len(encoding.input_ids),
          2,
          msg='option encoding error for %s' % c,
      )
    # and they should all be the same token.
    tok_2 = [enc.input_ids[1] for enc in p_encodings]
    are_all_periods = all(t == tok_2[0] for t in tok_2)
    self.assertTrue(are_all_periods)


# examples that represent all different tokens that can come before mask,
# and each color
_EXAMPLES = [
    'Most apples are [MASK].',
    'This apple is [MASK].',
]


class TestPreprocessSteps(parameterized.TestCase):
  """ test the full example preprocessing steps """

  @parameterized.named_parameters(
      ('albert-[xxx]', 'albert', create_example_albert),
      ('roberta-[xxx]', 'roberta', create_example_roberta),
      ('gpt-[xxx]', 'gpt2', create_example_gpt),
      ('gpt-loss-[xxx]', 'gpt2-loss', create_examples_gpt_loss),
  )
  def test_get_preprocessing_step_for_model(self, step_name, expected_step_fn):
    """ assert we use the correct step fn  """
    step_fn = get_preprocessing_step_for_model(step_name)
    self.assertEqual(step_fn, expected_step_fn)

  @parameterized.parameters(
      'albert-base-v1',
      'albert-base-v2',
      'albert-large-v1',
      'albert-large-v2',
      'albert-xlarge-v1',
      'albert-xlarge-v2',
      'albert-xxlarge-v1',
      'albert-xxlarge-v2',
  )
  def test_encode_example_albert(self, model_name):
    """ assert we process the examples correctly   """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    step_fn = get_preprocessing_step_for_model('albert')

    expected_texts = [
        '[CLS] most apples are blue.[SEP]',
        '[CLS] this apple is blue.[SEP]',
    ]

    for text, expected in zip(_EXAMPLES, expected_texts):
      instance = step_fn({'text': text}, tokenizer=tokenizer)
      # if we substitute in the label at the correct index, we should get the
      # expected decoding. this roughly corresponds to how we use target_tok_idx
      # in mlm_step.
      input_ids = instance['input_ids']
      input_ids[instance['target_tok_idx']] = instance['option_input_ids'][1]

      # decode and check for a text match.
      decoded = tokenizer.decode(input_ids)
      self.assertEqual(decoded, expected)

  @parameterized.parameters('roberta-base', 'roberta-large')
  def test_encode_example_roberta(self, model_name):
    """ assert we process the examples correctly   """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    step_fn = get_preprocessing_step_for_model('roberta')

    expected_texts = [
        '<s>Most apples are blue.</s>',
        '<s>This apple is blue.</s>',
    ]

    for text, expected in zip(_EXAMPLES, expected_texts):
      instance = step_fn({'text': text}, tokenizer=tokenizer)
      # if we substitute in the label at the correct index, we should get the
      # expected decoding. this roughly corresponds to how we use target_tok_idx
      # in mlm_step.
      input_ids = instance['input_ids']
      input_ids[instance['target_tok_idx']] = instance['option_input_ids'][1]

      # decode and check for a text match.
      decoded = tokenizer.decode(input_ids)
      self.assertEqual(decoded, expected)

  @parameterized.parameters('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
  def test_encode_example_gpt2(self, model_name):
    """ test gpt2 encoding"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    step_fn = get_preprocessing_step_for_model('gpt2')
    tokenizer.pad_token = tokenizer.unk_token

    expected_texts = [
        '<|endoftext|>Most apples are<|endoftext|><|endoftext|>',
        '<|endoftext|>This apple is<|endoftext|><|endoftext|>',
    ]

    for text, expected in zip(_EXAMPLES, expected_texts):
      instance = step_fn({'text': text}, tokenizer=tokenizer)
      # decode and check for a text match.
      decoded = tokenizer.decode(instance['input_ids'])
      self.assertEqual(decoded, expected)
      # check for the correct index.
      self.assertEqual(instance['target_tok_idx'], 3)


if __name__ == '__main__':
  absltest.main()
