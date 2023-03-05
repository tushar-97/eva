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
from typing import Iterator

import pandas as pd
from eva.executor.abstract_executor import AbstractExecutor
from eva.models.storage.batch import Batch
from eva.plan_nodes.train_plan import TrainPlan


class TrainExecutor(AbstractExecutor):
    def __init__(self, node: TrainPlan):
        super().__init__(node)
        self._func_expr = node.func_expr

    def validate(self):
        pass

    def exec(self) -> Iterator[Batch]:
        child_executor = self.children[0]
        aggregated_batch_list = []

        # aggregates the batches into one large batch
        for batch in child_executor.exec():
            aggregated_batch_list.append(batch)
        aggregated_batch = Batch.concat(aggregated_batch_list, copy=False)

        udf_name = self._func_expr.children[0].value
        func = self._func_expr._gpu_enabled_function()
        func(udf_name, aggregated_batch)

        yield Batch(
            pd.DataFrame(
                {f"UDF: {udf_name} successfully trained"},
                index=[0],
            )
        )
