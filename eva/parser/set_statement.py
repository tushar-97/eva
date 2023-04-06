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
from eva.expression.constant_value_expression import ConstantValueExpression
from eva.parser.statement import AbstractStatement
from eva.parser.types import StatementType


class SetStatement(AbstractStatement):
    def __init__(
        self,
        set_option: ConstantValueExpression,
        set_value: ConstantValueExpression,
    ):
        super().__init__(StatementType.SET)
        self._set_option = set_option
        self._set_value = set_value

    def __str__(self) -> str:
        set_str = f"SET {self._set_option} {self._set_value}"
        return set_str

    @property
    def set_option(self):
        return self._set_option

    @property
    def set_value(self):
        return self._set_value

    def __eq__(self, other):
        if not isinstance(other, SetStatement):
            return False
        return (
            self._set_option == other._set_option
            and self._set_value == other._set_value
        )

    def __hash__(self) -> int:
        return hash((super().__hash__(), self._set_option, self._set_value))
