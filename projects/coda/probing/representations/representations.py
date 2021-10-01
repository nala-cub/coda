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
""" zeroshot probing pipeline  """

from typing import List, Optional, Tuple

from absl import app
from absl import logging
from chex import clear_trace_counter
from ml_collections import FrozenConfigDict
from pandas import DataFrame
from pandas import concat

from labtools import catch_exp_failures
from probing._src.configurable import configurable
from probing.configs import get_configs
from probing.dataset.dataset import create_dataset
from probing.representations.build_reprs import build_repr_ds
from probing.representations.repr_probing import repr_probing
from probing.representations.repr_scoring import repr_scoring


def _repr_pipeline(
    configs: List[Tuple[str, FrozenConfigDict]],
    max_examples: Optional[int] = None,
) -> DataFrame:
  """ representation partial pipeline """

  result_csvs = []

  _, meta = create_dataset()

  # jax and torch don't play nice with memory - do torch first, then move on.
  # this will be better once with better flax support from hf.
  repr_ds_list = []
  for name, config in configs:
    repr_ds = build_repr_ds(model_name=config.model_name,
                            use_pretrained=config.use_pretrained,
                            max_examples=max_examples)
    repr_ds.reset_format()

    repr_ds_list.append(repr_ds)

  for (name, config), repr_ds in zip(configs, repr_ds_list):
    with catch_exp_failures(name):
      # clear chex counters from previous runs
      clear_trace_counter()
      repr_preds, _ = repr_probing(repr_ds=repr_ds,
                                   **config.repr,
                                   validation_split='test')
      # score preds
      results = repr_scoring(groundtruth=meta, preds=repr_preds)

      # results: add metadata cols.
      for c, v in config.report.items():
        results[c] = v

      # save outputs for later
      result_csvs.append(results)

      results.to_csv(f'{name}.csv')

  # collect zeroshot outputs.
  if len(result_csvs) > 0:
    df = concat(result_csvs)
    return df


@configurable('representations')
def _repr_pipeline_cmd(
    config: Optional[str] = None,
    savepath: Optional[str] = None,
) -> DataFrame:
  configs = get_configs(config, with_names=True)
  df = _repr_pipeline(configs=configs)
  if savepath and isinstance(df, DataFrame):
    df.to_csv(savepath, index=False)
  return df


def main(_):
  logging.get_absl_handler().use_absl_log_file('representations')
  _repr_pipeline_cmd()


if __name__ == '__main__':
  app.run(main)
