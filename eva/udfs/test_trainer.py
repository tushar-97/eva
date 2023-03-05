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
from datasets import Dataset
from eva.udfs.abstract.trainer_abstract_udf import AbstractTrainerUDF
from transformers import Trainer


class TestTrainer(AbstractTrainerUDF):
    @property
    def name(self) -> str:
        return "TestTrainer"

    def train(self, data):
        # Tokenize the sentences and create input_ids and attention_mask
        tokenized = self.tokenizer(
            data.column_as_numpy_array("data.data").tolist(),
            truncation=True,
            padding=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Create a Dataset object with input_ids, attention_mask, and labels
        dataset = Dataset.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": data.column_as_numpy_array("data.label").tolist(),
            }
        )

        # Split the dataset into training and validation sets
        dataset_dict = dataset.train_test_split(test_size=0.2)

        trainer = Trainer(
            model=self.model,  # the Hugging Face model
            args=self.default_training_args,  # training arguments
            train_dataset=dataset_dict["train"],  # training dataset
            eval_dataset=dataset_dict["test"],  # evaluation dataset
        )

        trainer.train()

        print(trainer.state.best_model_checkpoint)


# possible optimizations:
# early stopping if validation metric stops improving
# gradient accumulation to reduce memory usage when training large batches
# distributed training using accelerator
# padding to max not efficient, consult documentation
