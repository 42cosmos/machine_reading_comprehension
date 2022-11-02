from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from functools import partial


class Loader:
    def __init__(self, config, tokenizer):
        self.dset_name = config.dset_name
        self.task = config.task
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.stride = config.stride
        self.padding = config.padding
        self.truncation = config.truncation
        self.tokenizer = tokenizer

    def load(self, mode):
        dataset = load_dataset(self.dset_name, self.task, split=mode)

        if mode == "validation":
            dataset = dataset.map(
                self.preprocess_validation_examples,
                batched=True,
                remove_columns=dataset.column_names)

            dataset = dataset.remove_columns(["example_id", "offset_mapping"])

        else:
            dataset = dataset.map(
                self.preprocess_training_examples,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=False)

        dataset.set_format(type="torch")

        return dataset

    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation=self.truncation,
            stride=self.stride,
            padding=self.padding,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation=self.truncation,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["guid"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
