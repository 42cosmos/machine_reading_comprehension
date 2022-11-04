import yaml
import argparse
from typing import Any, Dict, List, Union

from easydict import EasyDict
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.data.processors.squad import SquadExample, squad_convert_examples_to_features


class KlueMRCExample(SquadExample):
    def __init__(self, question_type: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.question_type = question_type


class KlueMRCProcessor:
    def __init__(self):
        with open("config.yaml") as f:
            saved_hparams = yaml.load(f, Loader=yaml.FullLoader)
            self.hparams = EasyDict(saved_hparams)["CFG"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def get_train_dataset(self):
        return self._create_dataset("train")

    def get_test_dataset(self):
        return self._create_dataset("test")

    def _create_dataset(self, dataset_type: str):
        is_training = dataset_type == "train"
        examples = self._create_examples(is_training)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.hparams.max_seq_length,
            doc_stride=self.hparams.doc_stride,
            max_query_length=self.hparams.max_query_length,
            is_training=is_training,
            return_dataset="pt",
            threads=self.hparams.threads,
        )

        if not is_training:
            data = getattr(self.hparams, "data", {})
            data[dataset_type] = {"examples": examples, "features": features}
            setattr(self.hparams, "data", data)

        return dataset

    def _create_examples(self, is_training: bool = True):
        examples = []
        mode = "train" if is_training else "validation"
        data = load_dataset(self.hparams.dset_name, self.hparams.task, split=mode)
        for q in tqdm(data):
            context = q["context"]
            question = q["question"]
            question_type = q["question_type"]
            id_ = q["guid"]
            answer_text = q["answers"]["text"]
            answer_start = q["answers"]["answer_start"][0]
            answer_impossibility = q["is_impossible"]
            examples.append(
                KlueMRCExample(
                    question_type=question_type,
                    qas_id=id_,
                    question_text=question,
                    context_text=context,
                    answer_text=answer_text,
                    start_position_character=answer_start,
                    title="",
                    is_impossible=answer_impossibility,
                    answers=[],
                )
            )

        return examples
