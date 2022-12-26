#!/usr/bin/env bash
mkdir mrc_data
aws s3 sync s3://docent-information/mrc_data ./mrc_data --no-sign-request

python3 ./main.py \
        --do_train \
        --do_eval \

sed -i 's/dataset_name: klue/dataset_name: squad_v1_kor/g' config.yaml
python3 ./main.py \
        --do_train \
        --do_eval \
        --load_checkpoint

DOCENT_TRAIN_NUM=$(find ./mrc_data/docent/train -name "*.json" | wc -l)
DOCENT_VALID_NUM=$(find ./mrc_data/docent/validation -name "*.json" | wc -l)

echo "Klue and all datasets were trained !"
echo "And finetune with docent ( training set num: $DOCENT_TRAIN_NUM / valid set num: $DOCENT_VALID_NUM)"

sed -i 's/dataset_name: squad_v1_kor/dataset_name: docent/g' config.yaml
python3 ./main.py \
        --do_train \
        --do_eval \
        --load_checkpoint