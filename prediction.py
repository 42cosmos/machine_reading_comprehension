import argparse
import os
import json
import yaml
from easydict import EasyDict

from metrics import f1_by_character
from data_loader import MRCLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation')
    parser.add_argument('--prediction_file', help='Prediction File')  # output/predictions.json
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    loader = MRCLoader(config)
    examples, _ = loader.get_dataset(evaluate=True, output_examples=True)

    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    print(f1_by_character(examples=examples, predictions=predictions))
