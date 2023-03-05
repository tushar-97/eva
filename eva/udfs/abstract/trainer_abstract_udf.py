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
from abc import abstractmethod
from typing import Callable

from eva.udfs.abstract.abstract_udf import AbstractUDF, InputType
from eva.utils.trainer_utils import get_model_checkpoint_path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    TrainingArguments,
)


class AbstractTrainerUDF(AbstractUDF):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self._model = None
        self._tokenizer: Callable = lambda: None
        self._checkpoint_path = None
        self._default_training_args = TrainingArguments(
            output_dir="./results",  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_steps=10,
            load_best_model_at_end=True,
            save_total_limit=2,
            save_strategy="no",  # save a checkpoint only at the end of training
        )

    def __call__(self, *args, **kwargs):
        self._checkpoint_path = get_model_checkpoint_path(args[0])
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._checkpoint_path
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint_path)
        self.train(args[1])
        # TODO: look into saving best
        self._model.save_pretrained(self._checkpoint_path)

    @abstractmethod
    def train(self, data):
        pass

    def setup(self, *args, **kwargs) -> None:
        pass

    def forward(self, frames: InputType) -> InputType:
        pass

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> Callable:
        return self._tokenizer

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def default_training_args(self) -> TrainingArguments:
        return self._default_training_args
