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
from eva.parser.set_statement import SetStatement


##################################################################
# SET STATEMENTS
##################################################################
class Set:
    def set_statement(self, tree):
        set_list = self.visit_children(tree)
        assert len(set_list) == 3
        set_option = ConstantValueExpression(set_list[1])
        set_value = ConstantValueExpression(set_list[2])
        set_stmt = SetStatement(set_option, set_value)
        return set_stmt
