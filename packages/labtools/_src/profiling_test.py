# Copyright 2021 The LabTools Authors
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
""" Provides labtools profiling tests

Notes:
  To properly test the profiling module, it MUST NOT be imported at the
  toplevel. Doing so would add the default profiler init to the after_init
  callbacks. By importing the profiling module *after* import we ensure that
  `app.call_after_init(func)` is equivalent to `func()`
"""

from absl.testing import absltest
from absl.testing import parameterized


class TestProfiling(parameterized.TestCase):

  def test_profiling_flag_registration(self):
    """Tests that the profiling flag registers properly. """
    from labtools._src.profiling import \
        profiler  # pylint: disable=import-outside-toplevel

    # check default mode
    self.assertEqual(profiler.mode, 'disabled')


if __name__ == '__main__':
  absltest.main()
