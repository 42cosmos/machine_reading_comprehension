import os
import yaml
from easydict import EasyDict

from klue_data_loader import KlueMRCProcessor
from trainer import Trainer

import wandb
import logging
from dotenv import load_dotenv


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    load_dotenv()
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    with open("./config.yaml") as f:
        saved_hparams = yaml.load(f, Loader=yaml.FullLoader)
        hparams = EasyDict(saved_hparams)["CFG"]
    wandb.init(entity=hparams.entity_name, project=hparams.project_name, config=hparams)

    processor = KlueMRCProcessor()

    train_dataset = processor.get_dataset(evaluate=False, output_examples=False)
    test_dataset, examples, features = processor.get_dataset(evaluate=True, output_examples=True)

    trainer = Trainer(hparams,
                      train_dataset=train_dataset,
                      test_dataset=(test_dataset, examples, features)
                      )

    global_step, tr_loss = trainer.train()
    logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    trainer.evaluate()

    wandb.finish()
