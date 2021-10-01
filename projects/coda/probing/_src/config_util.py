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
""" Configuration Utilities """

from absl import flags
from absl import logging
from rules_python.python.runfiles import runfiles
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

FLAGS = flags.FLAGS


def hf_auto_configure(model_name, cache_dir=None, lazy=False):

  model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
  tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

  if model_config.model_type in ('openai-gpt', 'gpt2'):
    model_config.is_casual = True
    model_cls = AutoModelForCausalLM
    tokenizer.pad_token = tokenizer.unk_token
  else:
    model_cls = AutoModelForMaskedLM
    model_config.is_casual = False

  # get model
  model = lambda: model_cls.from_pretrained(model_name, cache_dir=cache_dir)
  model = model if lazy else model()

  return model_config, tokenizer, model


def maybe_rlocation(path: str) -> str:
  r = runfiles.Create()
  resolved_path = r.Rlocation(path)
  if resolved_path is None:
    logging.warning(
        'failed to resolve %s in runfiles tree. returning the original path.',
        path)
    resolved_path = path
  return resolved_path
