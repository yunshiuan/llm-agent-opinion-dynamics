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
parser.add_argument(
  "-reflection",
  "--reflection",
  action="store_true",
  help="Set flag if the memory is reflective (rather than cumulative)",
)
parser.add_argument(
    "-seed",
    "--seed",
    default="1",
    type=str,
    help="Seed of the result",
)
parser.add_argument(
    "-m",
    "--model_name",
    default="gpt-4-1106-preview",
    type=str,
    help="Name of the LLM to use as agents",
)
parser.add_argument(
  "-input_date",
    "--input_date",
    default="20231201",
    type=str,
    help="Date of the input file",
)
  
parser.add_argument(
    "-no_annotation",
    "--no_annotation",
    action="store_true",
    help="Set flag if no annotation is desired",
)
parser.add_argument(
  "-figure_file_type",
    "--figure_file_type",
    default="png",
    type=str,
    help="File type of the figure to be saved",
)

args = parser.parse_args()


prompt = args.prompt
prompt_root = "final_state_plots"
if args.reflection:
    file_prefix = "reflection_"
else:
    file_prefix = ""

# model abbreviation
if "gpt-4" in args.model_name:
  model_abbrev = "gpt-4_"
elif "gpt-3.5-turbo-16k" in args.model_name:
  model_abbrev = ""
else:
  raise ValueError("Model name not recognized")
    
out_name = (
    prompt_root
    + f"/{file_prefix}{model_abbrev}histogram_seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_{args.input_date}_flan-t5-xxl+{args.distribution}.{args.figure_file_type}"
)


main_name = (
    "final_csv_files"
    + f"/{file_prefix}{model_abbrev}seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_{args.input_date}_flan-t5-xxl_{args.distribution}.csv"
)

series = pd.read_csv(main_name)
print("Read in: ", main_name)
final_opinion_profile = series.iloc[:, 2:].iloc[100].values
bias = np.round(np.mean(final_opinion_profile), 2)
diversity = np.round(np.std(final_opinion_profile), 2)

if args.figure_file_type == "pdf":
    plt.figure(figsize=(3, 3))
elif args.figure_file_type == "png":
    plt.figure(figsize=(4, 3))
else:
    raise ValueError("Invalid file type for figure")

bin_edges = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

# title_list = [
#     "Default",
#     "Default + \nReverse Framing",
#     "Confirmation Bias",
#     "Confirmation Bias + \nReverse Framing",
# ]


plt.hist(final_opinion_profile, bins=bin_edges, color="grey")
plt.axvline(
    bias,
    color="darkred",
    linestyle="dotted",
    linewidth=3,
    label=f"Bias ({bias})",
)
if not args.no_annotation:    
    plt.xlabel("Opinion Profile")
    plt.ylabel("Number of Agents")
    plt.title(f"Diversity = {diversity}", fontsize=12)
    plt.legend(loc="upper right")
    plt.subplots_adjust(wspace=0.4)
    plt.tick_params(axis='both', which='major', labelsize=10)
else:
    plt.tick_params(axis='both', which='major', labelsize=14,width=2)
    # Increase border line width
    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_linewidth(2)
# x-axis ticks at -2, -1, 0, 1, 2
plt.xticks(np.arange(-2, 3, 1))  
# y axis ticks at 0, 2, 4, 6, 8, 10
plt.yticks(np.arange(0, 11, 2))  

plt.tight_layout()
if args.no_annotation:
    plt.tight_layout(pad=0.1)
plt.savefig(out_name)
print("Saved figure to: ", out_name)
