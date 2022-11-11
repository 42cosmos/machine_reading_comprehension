import argparse
import logging
import os
import sys
import yaml
from easydict import EasyDict

from trainer import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from data_loader import MRCLoader
from metrics import post_processing_function, compute_metrics
import wandb

logger = logging.getLogger(__name__)


def main(args):
    with open("config.yaml") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        hparams = EasyDict(saved_config)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    wandb.init(project=hparams.project_name, entity=hparams.entity_name)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(hparams.output_dir) and args.do_train:
        last_checkpoint = get_last_checkpoint(hparams.output_dir)
        if last_checkpoint is None and len(os.listdir(hparams.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({hparams.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and hparams.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    set_seed(hparams.seed)

    config = AutoConfig.from_pretrained(
        hparams.model_name_or_path,
        revision=hparams.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.model_name_or_path,
        revision=hparams.model_revision,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        hparams.model_name_or_path,
        revision=hparams.model_revision,
    )

    data_collator = (
        default_data_collator
        if hparams.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if hparams.fp16 else None)
    )

    loader = MRCLoader(hparams, tokenizer)

    train_dataset = loader.get_dataset(hparams.data_dir, evaluate=False, output_examples=False)
    eval_examples, eval_dataset = loader.get_dataset(hparams.data_dir, evaluate=True, output_examples=True)

    wandb.watch(model, log="all", log_freq=100)
    trainer = QuestionAnsweringTrainer(
        model=model,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        eval_examples=eval_examples if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if args.do_train:
        checkpoint = None
        if hparams.resume_from_checkpoint is not None:
            checkpoint = hparams.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        wandb.log(train_result)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": hparams.model_name_or_path, "tasks": "question-answering"}
    if hparams.dataset_name:
        kwargs["dataset_tags"] = hparams.dataset_name

    if hparams.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    args = parser.parse_args()

    main(args)
