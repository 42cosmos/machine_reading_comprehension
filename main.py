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

logger = logging.getLogger(__name__)


def main(args):
    with open("config.yaml") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        hparams = EasyDict(saved_config)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if args.load_checkpoint:
        last_checkpoint = None
        if os.path.exists(hparams.output_dir):
            last_checkpoint = get_last_checkpoint(hparams.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Checkpoint detected. Resuming from {last_checkpoint}")
                hparams.model_name_or_path = last_checkpoint
            else:
                logger.info(f"No checkpoint found, training from scratch: {hparams.model_name_or_path}")

        if not os.path.exists(hparams.output_dir):
            os.makedirs(hparams.output_dir)

    elif not args.load_checkpoint:
        if os.path.exists(hparams.output_dir) and len(os.listdir(hparams.output_dir)) > 1:
            hparams.model_name_or_path = hparams.output_dir
            logger.info(f"***** Load Model from {hparams.model_name_or_path} *****")

    set_seed(hparams.seed)

    training_args = TrainingArguments(
        output_dir=hparams.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=hparams.train_batch_size,
        per_device_eval_batch_size=hparams.eval_batch_size,
        learning_rate=float(hparams.learning_rate),
        adam_epsilon=float(hparams.adam_epsilon),
        num_train_epochs=hparams.num_train_epochs,
        weight_decay=hparams.weight_decay,
        logging_steps=hparams.logging_steps,
        seed=hparams.seed,
        fp16=hparams.fp16,
        warmup_steps=hparams.warmup_steps,
        max_steps=hparams.max_steps,
        log_level="info",
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

    loader = MRCLoader(hparams)

    if args.do_train:
        train_dataset = loader.get_dataset(evaluate=False, output_examples=False)

    if args.do_eval:
        eval_examples, eval_dataset = loader.get_dataset(evaluate=True, output_examples=True)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
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
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--load_checkpoint", action="store_true")

    args = parser.parse_args()

    main(args)
