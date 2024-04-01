import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Plotting Script")
parser.add_argument(
    "-agents",
    "--num_agents",
    default=5,
    type=int,
    help="Number of agents participating in the study",
)
parser.add_argument(
    "-steps",
    "--num_steps",
    default=5,
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
    "-p",
    "--prompt",
    default="v13_default",
    type=str,
    help="Name of the prompt version to plot",
)
args = parser.parse_args()

list_prompts = [
    f"{args.prompt}_default",
    f"{args.prompt}_default_reverse",
    f"{args.prompt}_confirmation_bias",
    f"{args.prompt}_confirmation_bias_reverse",
]

prompt = args.prompt
prompt_root = "results/opinion_dynamics/Flache_2017/gpt-3.5-turbo-16k/flan-t5-results"
out_name = (
    prompt_root
    + f"/histogram_seed1_{args.num_agents}_{args.num_steps}_{prompt}_20231031_flan-t5-xxl+{args.distribution}.png"
)

if prompt in ["v42", "v43", "v44", "v45", "v46"]:
    list_prompts = [
        f"{args.prompt}_default",
        f"{args.prompt}_confirmation_bias",
    ]

if args.distribution != "uniform":
    list_prompts = [
        f"{args.prompt}_default",
        f"{args.prompt}_default_reverse",
    ]

opinion_list, bias_list, diversity_list = [], [], []

for prompt in list_prompts:
    main_name = (
        prompt_root
        + f"/seed1_{args.num_agents}_{args.num_steps}_{prompt}_20231031_flan-t5-xxl_{args.distribution}.csv"
    )

    series = pd.read_csv(main_name)
    final_opinion_profile = series.iloc[:, 2:].iloc[100].values
    bias = np.round(np.mean(final_opinion_profile), 2)
    diversity = np.round(np.std(final_opinion_profile), 2)

    opinion_list.append(final_opinion_profile)
    bias_list.append(bias)
    diversity_list.append(diversity)

prompt = args.prompt
if args.distribution == "uniform":
    if prompt not in ["v42", "v43", "v44", "v45", "v46"]:
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))

        bin_edges = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

        title_list = [
            "Default",
            "Default + \nReverse Framing",
            "Confirmation Bias",
            "Confirmation Bias + \nReverse Framing",
        ]

        for i in range(4):
            axs[i].hist(opinion_list[i], bins=bin_edges, color="grey")
            axs[i].axvline(
                bias_list[i],
                color="darkred",
                linestyle="dotted",
                linewidth=3,
                label=f"Bias ({bias_list[i]})",
            )

            axs[i].set_xlabel("Opinion Profile")
            axs[i].set_ylabel("Number of Agents")
            axs[i].set_title(f"Diversity = {diversity_list[i]}", fontsize=12)
            axs[i].legend(loc="upper right")

        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.savefig(out_name)

    elif prompt in ["v42", "v43", "v44", "v45", "v46"]:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))

        bin_edges = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

        title_list = [
            "Default",
            "Confirmation Bias",
        ]

        for i in range(2):
            axs[i].hist(opinion_list[i], bins=bin_edges, color="grey")
            axs[i].axvline(
                bias_list[i],
                color="darkred",
                linestyle="dotted",
                linewidth=3,
                label=f"Bias ({bias_list[i]})",
            )

            axs[i].set_xlabel("Opinion Profile")
            axs[i].set_ylabel("Number of Agents")
            axs[i].set_title(f"{title_list[i]} \n(Diversity = {diversity_list[i]})", fontsize=12)
            axs[i].legend(loc="upper right")

        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.savefig(out_name)

else:
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    bin_edges = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    title_list = ["Default", "Default + \nReverse Framing"]

    for i in range(2):
        axs[i].hist(opinion_list[i], bins=bin_edges, color="grey")
        axs[i].axvline(
            bias_list[i],
            color="darkred",
            linestyle="dotted",
            linewidth=3,
            label=f"Bias ({bias_list[i]})",
        )

        axs[i].set_xlabel("Opinion Profile")
        axs[i].set_ylabel("Number of Agents")
        axs[i].set_title(f"Diversity = {diversity_list[i]}", fontsize=12)
        axs[i].legend(loc="upper right")

    plt.subplots_adjust(hspace=0.8)
    plt.tight_layout()
    plt.savefig(out_name)
