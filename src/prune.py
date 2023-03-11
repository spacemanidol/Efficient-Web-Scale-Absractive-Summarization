import copy
from torch import nn
import sys
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import torch
import pdb

layers = {
    "small": {
        1: [0],
        2: [0, 7],
        3: [0, 2, 5, 7],
        4: [0, 2, 4, 5, 7],
        5: [0, 2, 3, 4, 5, 7],
    },
    "base": {
        1: [0, 11],
        2: [0, 3, 8, 11],
        3: [0, 2, 4, 6, 8, 10],
        4: [0, 2, 4, 5, 6, 7, 8, 11],
        5: [0, 1, 2, 4, 5, 6, 7, 8, 10, 11],
    },
    "large": {
        1: [0, 5, 11, 23],
        2: [0, 1, 3, 7, 11, 15, 19, 23],
        3: [0, 2, 4, 6, 8, 12, 14, 16, 18, 20, 22, 23],
        4: [0, 1, 2, 4, 6, 7, 8, 12, 14, 15, 16, 18, 20, 21, 22, 23],
        5: [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    },
}


def main(model_path, output_path, model_type, encoder_ratio, decoder_ratio):
    print("Loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    struct_model = copy.deepcopy(model)
    print("Removing layers")
    if 6 > encoder_ratio:
        print("removing layers from encoder")
        oldModuleList = model.encoder.block
        newModuleList = nn.ModuleList()
        target_layers = len(layers[model_type][encoder_ratio])
        for i in range(0, target_layers):
            newModuleList.append(oldModuleList[layers[model_type][encoder_ratio][i]])
        struct_model.encoder.block = newModuleList
        struct_model.config.num_layers = len(newModuleList)
    if 6 > decoder_ratio:
        print("removing layers from decoder")
        oldModuleList = model.decoder.block
        newModuleList = nn.ModuleList()
        target_layers = len(layers[model_type][decoder_ratio])
        for i in range(0, target_layers):
            newModuleList.append(oldModuleList[layers[model_type][decoder_ratio][i]])
        struct_model.decoder.block = newModuleList
        struct_model.config.num_decoder_layers = len(newModuleList)
    print(struct_model)
    print("done removng layers. Saving Model")
    struct_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    print(
        "format: prune.py <model_path> <output_path> <model_type> <encoder ratio (out of 6)> <decoder ratio (out of 6)>>"
    )
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    model_type = sys.argv[3]
    encoder_layers = int(sys.argv[4])
    decoder_layers = int(sys.argv[5])
    main(model_path, output_path, model_type, encoder_layers, decoder_layers)
