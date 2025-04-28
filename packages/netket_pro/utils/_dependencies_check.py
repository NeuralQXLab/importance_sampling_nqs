# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains logic to check version of NetKet's dependencies and hard-error in
case of outdated dependencies that we know might break NetKet's numerical results
silently or unexpectedly.
"""

import importlib


# if not module_version("netket") >= (3, 11, 2):  # pragma: no cover
#    raise ImportError("You must update netket to the latest github version.")

if importlib.util.find_spec("netket.utils.timing") is None:
    raise ImportError("You must update netket to the latest github version.")
