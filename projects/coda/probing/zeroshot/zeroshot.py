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
from absl import flags
from absl import logging
from ml_collections import FrozenConfigDict
import pandas as pd
from pandas import DataFrame

from labtools import catch_exp_failures
from labtools import maybe_rlocation
from probing._src.configurable import configurable
from probing.configs import get_configs
from probing.dataset.dataset import create_dataset
from probing.zeroshot.run_probing import run_zeroshot
from probing.zeroshot.run_scoring import zeroshot_scoring


def _zeroshot_pipeline(
    configs: List[Tuple[str, FrozenConfigDict]],
    ngram_gbc_path: str,
    max_examples: Optional[int] = None,
) -> DataFrame:
  """ zeroshot partial pipeline """

  result_csvs = []

  _, meta = create_dataset()
  ngram_gbc_path = maybe_rlocation(ngram_gbc_path)
  ngrams = pd.read_csv(ngram_gbc_path)

  for name, config in configs:
    # skip non-zeroshot models (clip)
    if not config.zeroshot:
      continue
    with catch_exp_failures(name):
      zeroshot_preds = run_zeroshot(model_name=config.model_name,
                                    max_examples=max_examples)

      # score preds
      results = zeroshot_scoring(
          meta=meta,
          preds=zeroshot_preds,
          ngrams=ngrams,
      )
      # reset index for cat.
      results = results.reset_index()

      # results: add metadata cols.
      for c, v in config.report.items():
        results[c] = v

      # save zeroshot outputs for later
      result_csvs.append(results)

  # collect zeroshot outputs.
  if len(result_csvs) > 0:
    df = pd.concat(result_csvs)
    return df


@configurable('zeroshot')
def _zeroshot_pipeline_cmd(
    config: Optional[str] = None,
    savepath: Optional[str] = None,
    ngram_gbc_path: Optional[str] = None,
) -> DataFrame:
  configs = get_configs(config, with_names=True)
  df = _zeroshot_pipeline(configs=configs, ngram_gbc_path=ngram_gbc_path)
  if savepath and isinstance(df, DataFrame):
    df.to_csv(savepath)
  return df


def main(_):
  logging.get_absl_handler().use_absl_log_file('zeroshot')
  _zeroshot_pipeline_cmd()


if __name__ == '__main__':
  flags.mark_flag_as_required('zeroshot.ngram_gbc_path')
  app.run(main)
