import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import argparse
import os
import numpy as np
from os.path import exists, join

#######################
# Build argument parser
#######################
parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Plotting Script")
parser.add_argument(
    "-m",
    "--model_name",
    default="gpt-3.5-turbo-16k",
    type=str,
    help="Name of the LLM to use as agents",
)
parser.add_argument(
    "-steps",
    "--num_steps",
    default=5,
    type=int,
    help="Number of steps or pair samples in the experiment",
)
parser.add_argument(
    "-in", "--csv_name", type=str, required=True, help="Path to CSV file to visualize."
)
parser.add_argument("-test", "--test_run", action="store_true", help="Set flag if test run")
parser.add_argument(
    "-out", "--output_name", type=str, required=True, help="Name of the output file"
)
args = parser.parse_args()


if args.test_run:
    read_name = f"results/opinion_dynamics/Flache_2017/{args.model_name}/test_runs/{args.csv_name}"
    output_path = f"results/opinion_dynamics/Flache_2017/{args.model_name}/test_runs/plots/"
else:
    read_name = f"results/opinion_dynamics/Flache_2017/{args.model_name}/{args.csv_name}"
    output_path = f"results/opinion_dynamics/Flache_2017/{args.model_name}/plots/"

# If plots directory doesn't exist then create it
if not exists(output_path):
    os.makedirs(output_path)

output_path = join(output_path, args.output_name)

# Get data and extract time_steps and list of agents
data = pd.read_csv(read_name, index_col="Unnamed: 0")
x_axis = data["time_step"]
list_agents = list(data.columns)[1:]

list_opinions = []
for agent in list_agents:
    list_opinions.append(data[agent])
opinions = np.array(list_opinions)

time_points = np.arange(0, args.num_steps + 1)
opinion_colors = {2: "#0066FF", 1: "#66C2FF", 0: "#999999", -1: "#FF8080", -2: "#E60000"}
displacement = 0.03

plt.figure(figsize=(12, 6))

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
        time_points, jittered_opinions, label=f"Agent {i + 1}", color=opinion_colors[opinion_value]
    )

plt.xlabel("Time Steps")
plt.ylabel("Opinion Profile")
plt.title("Evolution of Agent Opinions")
plt.yticks([-2, -1, 0, 1, 2])
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig(output_path)
