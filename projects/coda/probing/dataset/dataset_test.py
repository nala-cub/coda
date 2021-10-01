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
""" Tests for create_dataset module """

from absl.testing import absltest
import pandas as pd

from probing.dataset.dataset import create_dataset


class TestCreateDataset(absltest.TestCase):
  """ Tests create dataset """

  def test_create_dataset_no_change(self):
    """ Tests creating the dataset is reproducible """
    ds, meta = create_dataset()

    ds2, meta2 = create_dataset()

    for split in ds:
      with self.subTest(f'datasets-{split}'):
        pd.testing.assert_frame_equal(ds[split].to_pandas(),
                                      ds2[split].to_pandas())

    with self.subTest('metadata'):
      pd.testing.assert_frame_equal(meta, meta2)

  def test_kmeans_affected_by_seed(self):
    """ tests for kmeans seeding stability """

    ds, meta = create_dataset(seed_for_kmeans=0)
    ds2, meta2 = create_dataset(seed_for_kmeans=9999)

    with self.subTest('metadata'):
      pd.testing.assert_frame_equal(meta, meta2)


if __name__ == '__main__':
  absltest.main()
