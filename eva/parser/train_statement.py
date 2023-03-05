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
from eva.expression.function_expression import FunctionExpression
from eva.parser.select_statement import SelectStatement
from eva.parser.statement import AbstractStatement
from eva.parser.types import StatementType


class TrainStatement(AbstractStatement):
    def __init__(self, func_expr: FunctionExpression, query: SelectStatement):
        super().__init__(StatementType.TRAIN)
        self._func_expr = func_expr
        self._query = query

    @property
    def func_expr(self):
        return self._func_expr

    @property
    def query(self):
        return self._query

    def __str__(self) -> str:
        return f"TRAIN WITH {self._func_expr} VALUES ({self._query})"

    def __eq__(self, other):
        if not isinstance(other, TrainStatement):
            return False
        return self._func_expr == other._func_expr and self._query == other._query

    def __hash__(self) -> int:
        return hash((super().__hash__(), self._func_expr, self._query))
