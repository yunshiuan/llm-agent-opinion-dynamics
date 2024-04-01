import os
import pandas as pd
import re
import csv
import argparse
import torch
from numpy.random import choice, shuffle
from datetime import date
from os.path import join
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig, BitsAndBytesConfig
from typing import Tuple

def initialize_model(model_path):
    global model, tokenizer

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model_config = AutoConfig.from_pretrained(
        model_path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code = True,
        config = model_config,
        # quantization_config = bnb_config,
        device_map = "auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code = True
    )

def main():
    initialize_model("lmsys/vicuna-13b-v1.5-16k")

    

if __name__ == "__main__":
    main()