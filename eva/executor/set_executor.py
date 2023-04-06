# coding=utf-8
# Copyright 2018-2022 EVA
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


import pandas as pd

from eva.configuration.configuration_manager import ConfigurationManager
from eva.executor.abstract_executor import AbstractExecutor
from eva.models.storage.batch import Batch
from eva.plan_nodes.set_plan import SetPlan
from eva.utils.logging_manager import logger


class SetExecutor(AbstractExecutor):
    CATEGORY_FOR_OPTION = {"BATCH_MEM_SIZE": "executor"}

    def __init__(self, node: SetPlan):
        super().__init__(node)
        self._set_option = node.set_option
        self._set_value = node.set_value

    def exec(self, *args, **kwargs):
        try:
            set_option = str(self._set_option)
            set_value = self._set_value.value
            ConfigurationManager().set_value(
                self.CATEGORY_FOR_OPTION[set_option], set_option, set_value
            )
            msg = f"Value: {self._set_option.value} successfully updated"
            logger.info(msg)
            yield Batch(
                pd.DataFrame(
                    {msg},
                    index=[0],
                )
            )
        except Exception as e:
            logger.error(str(e))
            yield Batch(
                pd.DataFrame(
                    {f"Failed to set value: {self._set_option.value}"},
                    index=[0],
                )
            )
