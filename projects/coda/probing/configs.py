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
""" all model pipeline configurations """
from __future__ import annotations

import re
from typing import Optional

from absl import logging
from ml_collections import ConfigDict
from ml_collections import FieldReference
from ml_collections import FrozenConfigDict
from ml_collections.config_dict import placeholder

from labtools import frozen


def get_known_config_builders(config_str: Optional[str] = None,
                              use_regex: bool = False):
  """ Get the known configurations that match a given config string.

  There are 2 special configurations:
    special/all: run all configs
    special/small: run all small configs. this is meant as a representative
      set of models (1 of each type spec) that can be used for development.
  Note that `None` is interpreted as special/all.
  """
  config_fns = [
      # special/small
      ('randinit-roberta', randinit_roberta),
      ('randinit-clip', randinit_clip),
      ('clip-vitb32', clip_vitb32),
      ('clip-rn50', clip_rn50),
      ('gpt2', gpt2),
      ('roberta-base', roberta_base),
      ('albert-base-v1', albert_base_v1),
      ('albert-base-v2', albert_base_v2),
      # --------------------------------
      ('clip-rn101', clip_rn101),
      ('clip-rn50x4', clip_rn50x4),
      ('gpt2-medium', gpt2_medium),
      ('gpt2-large', gpt2_large),
      ('roberta-large', roberta_large),
      ('albert-large-v1', albert_large_v1),
      ('albert-large-v2', albert_large_v2),
      ('gpt2-xl', gpt2_xl),
      ('albert-xlarge-v1', albert_xlarge_v1),
      ('albert-xlarge-v2', albert_xlarge_v2),
      ('albert-xxlarge-v1', albert_xxlarge_v1),
      ('albert-xxlarge-v2', albert_xxlarge_v2)
  ]
  # maybe filter
  if config_str is None or config_str == 'special/all':
    return config_fns
  if config_str == 'special/small':
    return config_fns[:8]

  def _match_fn(n: str) -> bool:
    return re.match(config_str, n) is not None if use_regex else config_str == n

  selected_configs = [(n, fn) for n, fn in config_fns if _match_fn(n)]

  if use_regex:
    logging.info('Selected %d/%d configs with regex pattern "%s"',
                 len(selected_configs), len(config_fns), config_str)
    logging.debug(list(selected_configs))

  return selected_configs


def get_config(config_str: str) -> FrozenConfigDict:
  config_fns = get_known_config_builders(config_str, False)
  return config_fns[config_str]()


def get_configs(config_str: str,
                use_regex: bool = True,
                with_names=False) -> list[FrozenConfigDict]:
  # get all matching config -> fn pairs
  config_fns = get_known_config_builders(config_str, use_regex)
  # build them
  if with_names:
    return [(name, fn()) for name, fn in config_fns]
  else:
    return [fn() for _, fn in config_fns]


#############
# Random init
#############


@frozen
def randinit_roberta():
  config = _roberta('roberta-base')
  config.report.model_type = 'Random'
  config.report.model_name = 'Random RoBERTa B'
  config.report.model_size = 'B'
  config.use_pretrained = False
  config.zeroshot = False
  return config


@frozen
def randinit_clip():
  config = _clip('ViT-B/32')
  config.report.model_type = 'Random'
  config.report.model_name = 'Random CLIP ViT-B/32'
  config.report.model_size = 'CLIP ViT-B/32'
  config.use_pretrained = False
  config.zeroshot = False
  return config


###########
# CLIP
###########


@frozen
def clip_vitb32():
  config = _clip('ViT-B/32')
  return config


@frozen
def clip_rn50():
  config = _clip('RN50')
  return config


@frozen
def clip_rn50x4():
  config = _clip('RN50x4')
  return config


@frozen
def clip_rn101():
  config = _clip('RN101')
  return config


def _clip(model_name: str):
  config = _config_base(model_name)
  config.use_pretrained = True
  # no zeroshot
  config.zeroshot = False
  # in the case of clip, model_name is the same as reporting.
  config.report.model_type = 'CLIP'
  config.report.model_name = f'CLIP {model_name}'
  config.report.model_size = model_name

  return config


###########
# GPT 2
###########


@frozen
def gpt2():
  config = _gpt('gpt2')
  config.report.model_name = 'GPT2'
  config.report.model_size = 'B'
  return config


@frozen
def gpt2_medium():
  config = _gpt('gpt2-medium')
  config.report.model_name = 'GPT2 M'
  config.report.model_size = 'M'
  return config


@frozen
def gpt2_large():
  config = _gpt('gpt2-large')
  config.report.model_name = 'GPT2 L'
  config.report.model_size = 'L'
  return config


@frozen
def gpt2_xl():
  config = _gpt('gpt2-xl')
  config.report.model_name = 'GPT2 XL'
  config.report.model_size = 'XL'
  return config


def _gpt(model_name: str):
  config = _config_base(model_name)
  config.use_pretrained = True
  config.zeroshot = True
  config.report.model_type = 'GPT2'
  return config


###########
# RoBERTa
###########


@frozen
def roberta_base():
  config = _roberta('roberta-base')
  config.report.model_name = 'RoBERTa B'
  config.report.model_size = 'B'
  return config


@frozen
def roberta_large():
  config = _roberta('roberta-large')
  config.report.model_name = 'RoBERTa L'
  config.report.model_size = 'L'
  return config


def _roberta(model_name: str):
  config = _config_base(model_name)
  config.use_pretrained = True
  config.zeroshot = True
  config.report.model_type = 'RoBERTa'
  return config


###########
# ALBERT
###########


@frozen
def albert_base_v1():
  config = _albert('albert-base-v1')
  config.report.model_name = 'ALBERT V1 B'
  config.report.model_size = 'B'
  return config


@frozen
def albert_large_v1():
  config = _albert('albert-large-v1')
  config.report.model_name = 'ALBERT V1 L'
  config.report.model_size = 'L'
  return config


@frozen
def albert_xlarge_v1():
  config = _albert('albert-xlarge-v1')
  config.report.model_name = 'ALBERT V1 XL'
  config.report.model_size = 'XL'
  return config


@frozen
def albert_xxlarge_v1():
  config = _albert('albert-xxlarge-v1')
  config.report.model_name = 'ALBERT V1 XXL'
  config.report.model_size = 'XXL'
  return config


@frozen
def albert_base_v2():
  config = _albert('albert-base-v2')
  config.report.model_name = 'ALBERT V2 B'
  config.report.model_size = 'B'
  return config


@frozen
def albert_large_v2():
  config = _albert('albert-large-v2')
  config.report.model_name = 'ALBERT V2 L'
  config.report.model_size = 'L'
  return config


@frozen
def albert_xlarge_v2():
  config = _albert('albert-xlarge-v2')
  config.report.model_name = 'ALBERT V2 XL'
  config.report.model_size = 'XL'
  return config


@frozen
def albert_xxlarge_v2():
  config = _albert('albert-xxlarge-v2')
  config.report.model_name = 'ALBERT V2 XXL'
  config.report.model_size = 'XXL'
  return config


def _albert(model_name: str):
  config = _config_base(model_name)
  config.use_pretrained = True
  config.zeroshot = True
  config.report.model_type = 'ALBERT'
  return config


def _config_base(model_name) -> ConfigDict:
  batch_size = int(64)
  batch_size = FieldReference(batch_size)

  nb_seeds = FieldReference(5)
  nb_points = FieldReference(10)
  model_name = FieldReference(model_name)

  # general
  config = ConfigDict()
  config.model_name = model_name
  config.use_pretrained = placeholder(bool)
  config.zeroshot = placeholder(bool)

  # representations
  config.repr = ConfigDict()
  config.repr.nb_seeds = nb_seeds
  config.repr.nb_points = nb_points
  config.repr.max_parallel = -1

  # report
  config.report = ConfigDict()
  config.report.model_name = placeholder(str)
  config.report.model_type = placeholder(str)
  config.report.model_size = placeholder(str)

  return config
