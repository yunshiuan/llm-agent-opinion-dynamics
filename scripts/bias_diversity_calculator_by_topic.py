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
    default="gpt-3.5-turbo-16k",
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
    "-pv",
    "--prompt_versions",
    nargs='+', 
    help='All prompt versions', 
    required=True
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

prompt_bases = ["default", "confirmation_bias", "strong_confirmation_bias",
                "default_reverse", "confirmation_bias_reverse", "strong_confirmation_bias_reverse"]
# prompt_versions = [37, 38, 39, 40, 41, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
prompt_versions = args.prompt_versions


for base in prompt_bases:
    if "reverse" in base:
        continue

    for version in prompt_versions:
        prompt = f"v{version}_{base}"
        out_name = f"topic_results/v{prompt}.txt"

        bias_list, diversity_list = [], []
        with open(out_name, "a") as f:
            f.write(f"{base}: ")
            main_name = (
                prompt_root
                + f"/seed1_{args.num_agents}_{args.num_steps}_{prompt}_20231201_flan-t5-xxl_{args.distribution}.csv"
            )
            if args.reflection:
                main_name = (
                prompt_root
                + f"/reflection_seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_20231201_flan-t5-xxl_{args.distribution}.csv"
            )

            series = pd.read_csv(main_name)
            # get the final opinion profile (T=100)
            final_opinion_profile = series.iloc[:, 2:].iloc[100].values
            bias = np.round(np.mean(final_opinion_profile), 2)
            diversity = np.round(np.std(final_opinion_profile), 2)
            bias_list.append(bias)
            diversity_list.append(diversity)

    f.write(f"Bias: {np.round(np.mean(bias_list), 2)} $\pm$ {np.round(np.std(bias_list), 2)}\t, ")
    f.write(f"Diversity: {np.round(np.mean(diversity_list), 2)} $\pm$ {np.round(np.std(diversity_list), 2)}\n")

for base in prompt_bases:
    if "reverse" not in base:
        continue
    f.write(f"{base}: ")
    bias_list, diversity_list = [], []
    for version in prompt_versions:
        prompt = f"v{version}_{base}"
        # f.write("\n" + prompt + " : ")
        main_name = (
            prompt_root
            + f"/seed1_{args.num_agents}_{args.num_steps}_{prompt}_20231201_flan-t5-xxl_{args.distribution}.csv"
        )
        if args.reflection:
            main_name = (
            prompt_root
            + f"/reflection_seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_20231201_flan-t5-xxl_{args.distribution}.csv"
        )

        series = pd.read_csv(main_name)
        final_opinion_profile = series.iloc[:, 2:].iloc[100].values
        bias = np.round(np.mean(final_opinion_profile), 2)
        diversity = np.round(np.std(final_opinion_profile), 2)
        bias_list.append(bias)
        diversity_list.append(diversity)

    f.write(f"Bias: {np.round(np.mean(bias_list), 2)} $\pm$ {np.round(np.std(bias_list), 2)}\t, ")
    f.write(f"Diversity: {np.round(np.mean(diversity_list), 2)} $\pm$ {np.round(np.std(diversity_list), 2)}\n")
