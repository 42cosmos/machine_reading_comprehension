#!/usr/bin/env bash
mkdir mrc_data
aws s3 sync s3://{my_bucket}/{my_obj_name} ./mrc_data --no-sign-request

python3 main.py \
        --do_train \
        --do_eval \
        --dset_name klue

python3 main.py \
        --do_train \
        --do_eval \
        --dset_name squad_v1_kor \
        --load_checkpoint

DOCENT_TRAIN_NUM=$(find ./mrc_data/docent/train -name "*.json" | wc -l)
DOCENT_VALID_NUM=$(find ./mrc_data/docent/validation -name "*.json" | wc -l)

python main.py \
        --do_train \
        --do_eval \
        --load_checkpoint \
        --dset_name docent