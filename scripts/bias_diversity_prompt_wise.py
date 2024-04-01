import os
import pandas as pd
import re
import csv
import argparse
import numpy as np
from numpy.random import choice, shuffle
from datetime import date
from os.path import join
from collections import defaultdict
from typing import Tuple

parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Script")
parser.add_argument(
    "-m",
    "--model_name",
    default="gpt-4-1106-preview",
    type=str,
    help="Name of the LLM to use as agents",
)
parser.add_argument(
    "--reflection", action="store_true", help="Set flag if reflection prompt is being used"
)
parser.add_argument(
    "-agents",
    "--num_agents",
    default=10,
    type=int,
    help="Number of agents participating in the study",
)
parser.add_argument(
    "-steps",
    "--num_steps",
    default=100,
    type=int,
    help="Number of steps or pair samples in the experiment",
)
parser.add_argument(
    "-dist",
    "--distribution",
    default="uniform",
    choices=["uniform", "skewed_positive", "skewed_negative", "positive", "negative"],
    type=str,
    help="Type of initial opinion distribution",
)
parser.add_argument(
    "-seed",
    "--seed",
    default=1,
    type=int,
    help="Set reproducibility seed",
)
args = parser.parse_args()

prompt_root = f"final_csv_files"

out_name = "reflective_metrics_gpt_4.txt"
# out_name = "all_prompts_cumulative.txt"


prompt_bases = ["default", "confirmation_bias", "strong_confirmation_bias",
                "default_reverse", "confirmation_bias_reverse", "strong_confirmation_bias_reverse"]
prompt_versions = [37, 38, 39, 40, 41,]

with open(out_name, "a") as f:
    for version in prompt_versions:
        for base in prompt_bases:
            prompt = f"v{version}_{base}"
            f.write(f"{prompt}:\t")
            main_name = (
                prompt_root
                + f"/gpt-4_seed1_{args.num_agents}_{args.num_steps}_{prompt}_20231201_flan-t5-xxl_{args.distribution}.csv"
            )
            if args.reflection:
                main_name = (
                prompt_root
                + f"/reflection_gpt-4_seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_20231201_flan-t5-xxl_{args.distribution}.csv"
            )

            series = pd.read_csv(main_name)
            final_opinion_profile = series.iloc[:, 2:].iloc[100].values
            bias = np.round(np.mean(final_opinion_profile), 2)
            diversity = np.round(np.std(final_opinion_profile), 2)

            f.write(f"{bias} +- {diversity}\n")
