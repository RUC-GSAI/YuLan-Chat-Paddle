# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from ..llama.configuration import *
from .modeling import *
from ..llama.modeling_auto import *

try:
    from ..llama.modeling_auto_pp import *
except (ImportError, ModuleNotFoundError):
    # Temporarily adapt to the release version of Paddle, which can be removed later.
    pass
from ..llama.modeling_network import *
from ..llama.modeling_pp import *
from ..llama.tokenizer import *
from ..llama.tokenizer_fast import *
