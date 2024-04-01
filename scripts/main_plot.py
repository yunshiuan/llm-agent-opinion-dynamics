import copy
import pickle
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Argument Parser for Opinion Dynamics Plotting Script")
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
    choices=["uniform", "skewed_positive",
             "skewed_negative", "positive", "negative"],
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
    help="Date of the input pickle file",
)
parser.add_argument(
    "-input_csv_date",
    "--input_csv_date",
    default=None,
    type=str,
    help="Date of the input csv file. May be different from the input pickle file date",
)
# auto find the csv file with the correct date
parser.add_argument(
    "-auto_find_csv_date",
    "--auto_find_csv_date",
    action="store_true",
    help="Set flag if the csv file with the correct date should be automatically found. If set, the input_csv_date argument will be ignored.",
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
parser.add_argument(
    "-constant_line",
    "--constant_line",
    action="store_true",
    help="Set flag if a constant line is desired. The promt will be ignored.",
)
parser.add_argument(
    "-constant_value",
    "--constant_value",
    default=None,
    type=int,
    help="Value of the constant line (if `-constant_line`)",
)

args = parser.parse_args()

prompt = args.prompt
if args.reflection:
  file_prefix = "reflection_"
  file_suffix = "_reflection"
else:
  file_prefix = ""
  file_suffix = ""

if args.input_csv_date is None:
  args.input_csv_date = args.input_date

if "control" not in prompt:
  response_or_tweet = "response"
else:
  # control prompt result has a different naming convention
  response_or_tweet = "tweet"

# model abbreviation
if "gpt-4" in args.model_name:
  model_abbrev = "gpt-4_"
elif "gpt-3.5-turbo-16k" in args.model_name:
  model_abbrev = ""
elif "vicuna-33b-v1.3" in args.model_name:
  model_abbrev = "vicuna_"
else:
  raise ValueError("Model name not recognized")

main_name = (
    f"{model_abbrev}seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_{args.input_date}_flan-t5-xxl_{args.distribution}"
)
in_pkl_name = (
    f"results/opinion_dynamics/Flache_2017/{args.model_name}/flan-t5-results/{main_name}{file_suffix}.pkl"
)
in_name = f"final_csv_files/{file_prefix}{main_name}.csv"

if not args.constant_line:
  out_name = (
      f"final_plots/{file_prefix}{model_abbrev}seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_{args.input_date}_flan-t5-xxl_{args.distribution}.{args.figure_file_type}"
  )
else:
  out_name = (
      f"final_plots/line_constant_{args.constant_value}.{args.figure_file_type}"
  )
if not args.constant_line:
  with open(in_pkl_name, "rb") as f:
    results_pickle = pickle.load(f)
  results = copy.deepcopy(results_pickle)

  if args.auto_find_csv_date:
    # use regex to find the csv file with the correct date
    import glob
    import re
    # import pdb
    # pdb.set_trace()
    all_csv_files = glob.glob(
        f"results/opinion_dynamics/Flache_2017/{args.model_name}/seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_*_agent_{response_or_tweet}_history_{args.distribution}{file_suffix}.csv")      
    regex_pattern = f"^results/opinion_dynamics/Flache_2017/{args.model_name}/seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_(\d{{8}})_agent_{response_or_tweet}_history_{args.distribution}{file_suffix}\.csv$"

    matched_csv_dates = []
    for csv_file in all_csv_files:
      match = re.match(regex_pattern, csv_file)
      if match:
        # Extract the first capturing group, which is the eight-digit date
        matched_csv_dates.append(match.group(1))

    # pdb.set_trace()
    matched_csv_dates.sort()
    # throw a warning when there are multiple csv files with different dates, and print them out
    if len(matched_csv_dates) > 1:
      print(
          f"Warning: multiple csv files are found for prompt {prompt} and distribution {args.distribution}")
      print(f"Found csv dates: {matched_csv_dates}")
    elif len(matched_csv_dates) == 1:
      print(f"Found csv date: {matched_csv_dates[0]}")
    else:
      # import pdb
      # pdb.set_trace()
      # regex_pattern
      print("Attempting to find the csv file with pattern: ", regex_pattern)
      raise ValueError(
          f"No csv file is found for prompt {prompt} and distribution {args.distribution}")
    input_csv_date = str(matched_csv_dates[-1])
  else:
    input_csv_date = args.input_csv_date

  file_csv = f"results/opinion_dynamics/Flache_2017/{args.model_name}/seed{args.seed}_{args.num_agents}_{args.num_steps}_{prompt}_{input_csv_date}_agent_{response_or_tweet}_history_{args.distribution}{file_suffix}.csv"
  print("Reading from csv file: ", file_csv)

  series = pd.read_csv(file_csv)

  new_ts_df = {}
  new_ts_df["time_step"] = list(range(args.num_steps + 1))
  for agent in results:
    agent_df = series[series["Agent Name"] == agent].reset_index()
    new_ts_df[agent] = [agent_df["Original Belief"].values[0]]
    belief_change_steps = literal_eval(agent_df["Belief Changes Time Step"][0])
    for time_step in range(1, args.num_steps + 1):
      if time_step not in belief_change_steps:
        new_ts_df[agent].append(new_ts_df[agent][-1])
      else:
        new_ts_df[agent].append(results[agent]["Agent_Belief"].pop(0))
  pd.DataFrame.from_dict(new_ts_df).to_csv(in_name)

  data = pd.read_csv(in_name, index_col="Unnamed: 0")
  x_axis = data["time_step"]
  list_agents = list(data.columns)[1:]

  list_opinions = []
  for agent in list_agents:
    list_opinions.append(data[agent])
  opinions = np.array(list_opinions)

time_points = np.arange(0, args.num_steps + 1)
opinion_colors = {2: "#0066FF", 1: "#66C2FF",
                  0: "#999999", -1: "#FF8080", -2: "#E60000"}
displacement = 0.03

if args.figure_file_type == "pdf":
  plt.figure(figsize=(4.7, 3))
elif args.figure_file_type == "png":
  plt.figure(figsize=(4, 3))
else:
  raise ValueError("Invalid file type for figure")

# ------------
# Plot a constant line
# ------------
if args.constant_line:
  assert args.constant_value in opinion_colors, f"Constant value {args.constant_value} is not in opinion_colors"
  color = opinion_colors[int(args.constant_value)]

  plt.figure(figsize=(4.7, 3) if args.figure_file_type == "pdf" else (4, 3))
  plt.axhline(y=args.constant_value, color=color, linewidth=2)
  plt.ylim(-2.5, 2.5)  # Adjust as needed
  plt.xlim(0, args.num_steps)  # Set x-axis limits
else:
  # ------------
  # Plot the opinion evolution
  # ------------
  for i in range(len(opinions)):
    opinion_value = opinions[i, 0]
    jittered_opinions = [opinion_value + i * displacement]
    for j in range(1, len(time_points)):
      if opinions[i, j] != opinions[i, j - 1]:
        opinion_value = opinions[i, j]
        jittered_opinions.append(opinion_value + i * displacement)
      else:
        jittered_opinions.append(jittered_opinions[-1])
    opinion_value = opinions[i, 0]
    plt.plot(
        time_points, jittered_opinions, label=f"Agent {i + 1}", color=opinion_colors[opinion_value],
        linewidth=2
    )

y_ticks = [-2, -1, 0, 1, 2]
if not args.no_annotation:
  plt.xlabel("Time Steps")
  plt.ylabel("Opinion Profile")
  plt.title("Evolution of Agent Opinions")
  plt.yticks(y_ticks)
  plt.grid(True, linestyle="--")
  plt.tick_params(axis='both', which='major', labelsize=10)
else:
  plt.tick_params(axis='both', which='major', labelsize=14, width=2)
  plt.yticks(y_ticks)
  # Increase border line width
  ax = plt.gca()  # Get current axes
  for spine in ax.spines.values():
    spine.set_linewidth(2)
  # # Add horizontal lines at each y-tick
  # for y in y_ticks:
  #     plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)  # Adjust color, linestyle, and linewidth as needed

plt.tight_layout()
if args.no_annotation:
  plt.tight_layout(pad=0.1)
plt.savefig(out_name)
