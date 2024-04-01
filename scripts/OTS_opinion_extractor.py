from ast import literal_eval
import pickle
import torch
import argparse
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Script")
parser.add_argument(
    "-m",
    "--model_name",
    default="gpt-3.5-turbo-16k",
    type=str,
    help="Name of the LLM to use as agents",
)
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
    "-version",
    "--version_set",
    default="v5_default",
    type=str,
    help="Prompt version directory to use",
)
parser.add_argument(
    "-date",
    "--date_input",
    default="20231212",
    type=str,
    help="Input Date to Use",
)
parser.add_argument(
    "-s",
    "--seed",
    default=1,
    type=int,
    help="Seed for run",
)
parser.add_argument("-test", "--test_run", action="store_true", help="Set flag if test run")
parser.add_argument("-reflect", "--reflection", action="store_true", help="Set flag if reflective memory")
parser.add_argument("--model", default="google/flan-t5-xxl", help="Open Source Model to Use")
parser.add_argument("--model_dir", default="model/", help="Open Source Model to Use")
parser.add_argument(
    "-out", "--output_file", type=str, default="flan-t5-xxl", help="Name of the output file"
)
args = parser.parse_args()

tokenizer = T5Tokenizer.from_pretrained(
    args.model,
    cache_dir=args.model_dir,
)
model = T5ForConditionalGeneration.from_pretrained(
    args.model,
    cache_dir=args.model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

if args.test_run:
    print("Torch version:", torch.__version__)
    print("Is CUDA enabled?", torch.cuda.is_available())

if args.reflection:
    suffix = "_reflection"
else:
    suffix = ""


if args.version_set in ["v13_default", "v14_confirmation_bias", "v14_confirmation_bias_each_time"]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that "promotes vaccination with the vaccine VAX" right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about Bill XYZ), -1: (Slightly negative opinion about Bill XYZ), 0: (Neutral opinion about the Bill XYZ),
1: (Slightly positive opinion about Bill XYZ), 2: (Strongly positive opinion about Bill XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v15_default",
    "v15_default_no_rating",
    "v16_confirmation_bias",
    "v16_confirmation_bias_each_time",
    "v16_confirmation_bias_each_time_no_rating",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that "opposes vaccination with the vaccine VAX" right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about Bill XYZ), -1: (Slightly negative opinion about Bill XYZ), 0: (Neutral opinion about the Bill XYZ),
1: (Slightly positive opinion about Bill XYZ), 2: (Strongly positive opinion about Bill XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v17_default",
    "v18_confirmation_bias",
    "v18_confirmation_bias_each_time",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that "promotes the rights of gun ownership" right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about Bill XYZ), -1: (Slightly negative opinion about Bill XYZ), 0: (Neutral opinion about the Bill XYZ),
1: (Slightly positive opinion about Bill XYZ), 2: (Strongly positive opinion about Bill XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v19_default",
    "v20_confirmation_bias",
    "v20_confirmation_bias_each_time",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that "opposes the rights of gun ownership" right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about Bill XYZ), -1: (Slightly negative opinion about Bill XYZ), 0: (Neutral opinion about the Bill XYZ),
1: (Slightly positive opinion about Bill XYZ), 2: (Strongly positive opinion about Bill XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v21_default",
    "v22_confirmation_bias",
    "v22_confirmation_bias_each_time",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about a movie XYZ right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about movie XYZ), -1: (Slightly negative opinion about movie XYZ), 0: (Neutral opinion about the movie XYZ),
1: (Slightly positive opinion about movie XYZ), 2: (Strongly positive opinion about movie XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v23_default",
    "v23_default_closed_world",
    "v23_default_closed_world_no_rating",
    "v24_confirmation_bias",
    "v24_confirmation_bias_each_time",
    "v24_confirmation_bias_each_time_closed_world_no_rating",
    "v24_repulsion_each_time",
    "v24_confirmation_bias_reinforcement_each_time",
    "v24_confirmation_bias_reinforcement_each_time_closed_world_no_rating",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the XYZ fast food restaurant right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about fast food restaurant XYZ), -1: (Slightly negative opinion about fast food restaurant XYZ), 
0: (Neutral opinion about the fast food restaurant XYZ), 1: (Slightly positive opinion about fast food restaurant XYZ), 
2: (Strongly positive opinion about fast food restaurant XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v25_default",
    "v26_confirmation_bias",
    "v26_confirmation_bias_each_time",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the theory XYZ that claims that the earth is flat right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about theory XYZ), -1: (Slightly negative opinion about theory XYZ), 
0: (Neutral opinion about theory XYZ), 1: (Slightly positive opinion about theory XYZ), 
2: (Strongly positive opinion about theory XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v27_default",
    "v28_confirmation_bias",
    "v28_confirmation_bias_each_time",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the theory XYZ that claims that the earth is an irregularly shaped ellipsoid rather than flat, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about theory XYZ), -1: (Slightly negative opinion about theory XYZ), 
0: (Neutral opinion about theory XYZ), 1: (Slightly positive opinion about theory XYZ), 
2: (Strongly positive opinion about theory XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v29_default",
    "v29_default_closed_world_no_rating",
    "v30_confirmation_bias",
    "v30_confirmation_bias_each_time",
    "v30_confirmation_bias_each_time_closed_world_no_rating",
    "v30_confirmation_bias_reinforcement_each_time_closed_world_no_rating",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the mystery novel XYZ right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about novel XYZ), -1: (Slightly negative opinion about novel XYZ), 
0: (Neutral opinion about novel XYZ), 1: (Slightly positive opinion about novel XYZ), 
2: (Strongly positive opinion about novel XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v31_default",
    "v31_default_closed_world_no_rating",
    "v32_confirmation_bias",
    "v32_confirmation_bias_each_time",
    "v32_confirmation_bias_each_time_closed_world_no_rating",
    "v32_confirmation_bias_reinforcement_each_time_closed_world_no_rating",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the jazz band XYZ right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about band XYZ), -1: (Slightly negative opinion about band XYZ), 
0: (Neutral opinion about band XYZ), 1: (Slightly positive opinion about band XYZ), 
2: (Strongly positive opinion about band XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v33_default",
    "v34_confirmation_bias",
    "v34_confirmation_bias_each_time",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that "promotes" the use of VR headsets in high school education right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about Bill XYZ), -1: (Slightly negative opinion about Bill XYZ), 0: (Neutral opinion about the Bill XYZ),
1: (Slightly positive opinion about Bill XYZ), 2: (Strongly positive opinion about Bill XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v35_default",
    "v36_confirmation_bias",
    "v36_confirmation_bias_each_time",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that "opposes" the use of VR headsets in high school education right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about Bill XYZ), -1: (Slightly negative opinion about Bill XYZ), 0: (Neutral opinion about the Bill XYZ),
1: (Slightly positive opinion about Bill XYZ), 2: (Strongly positive opinion about Bill XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

######################################### New Prompt Versions #########################################

elif args.version_set in [
    "v37_default",
    "v37_confirmation_bias",
    "v37_strong_confirmation_bias",
    "v37_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the Earth is flat, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v37_default_reverse",
    "v37_confirmation_bias_reverse",
    "v37_strong_confirmation_bias_reverse",
    "v37_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the Earth is an irregularly-shaped ellipsoid rather than flat, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v38_default",
    "v38_confirmation_bias",
    "v38_strong_confirmation_bias",
    "v38_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the Tyrannosaurus Rex and humans co-existed on Earth at the same time, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v38_default_reverse",
    "v38_confirmation_bias_reverse",
    "v38_strong_confirmation_bias_reverse",
    "v38_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the Tyrannosaurus Rex and humans did not co-exist on Earth at the same time, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v39_default",
    "v39_confirmation_bias",
    "v39_strong_confirmation_bias",
    "v39_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that it is possible for humans to communicate with the dead, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v39_default_reverse",
    "v39_confirmation_bias_reverse",
    "v39_strong_confirmation_bias_reverse",
    "v39_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that it is not possible for humans to communicate with the dead, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v40_default",
    "v40_confirmation_bias",
    "v40_strong_confirmation_bias",
    "v40_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that it is possible to predict someone's future by looking at their palm characteristics, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v40_default_reverse",
    "v40_confirmation_bias_reverse",
    "v40_strong_confirmation_bias_reverse",
    "v40_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that it is not possible to predict someone's future by looking at their palm characteristics, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v41_default",
    "v41_confirmation_bias",
    "v41_strong_confirmation_bias",
    "v41_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that global warming is a conspiracy by governments worldwide and is not a real phenomenon, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v41_default_reverse",
    "v41_confirmation_bias_reverse",
    "v41_strong_confirmation_bias_reverse",
    "v41_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that global warming is a real phenomenon and global climate is rapidly growing warmer, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v42_default",
    "v42_confirmation_bias",
    "v42_strong_confirmation_bias",
    "control_v4_fastfood",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the fast food restaurant XYZ, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v43_default",
    "v43_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the mystery novel XYZ, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v44_default",
    "v44_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the jazz band XYZ, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v45_default",
    "v45_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the sci-fi movie XYZ, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v46_default",
    "v46_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the raisin and walnut oatmeal XYZ, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v47_default",
    "v47_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes increased public access to firearms, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v47_default_reverse",
    "v47_confirmation_bias_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes restricted public access to firearms, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v48_default",
    "v48_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes vaccination with the vaccine VAX, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v48_default_reverse",
    "v48_confirmation_bias_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that opposes vaccination with the vaccine VAX, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v49_default",
    "v49_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes increased public access to marijuana, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v49_default_reverse",
    "v49_confirmation_bias_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes restricted public access to marijuana, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v50_default",
    "v50_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes increased public access to euthanasia clinics, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v50_default_reverse",
    "v50_confirmation_bias_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes restricted public access to euthanasia clinics, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v51_default",
    "v51_confirmation_bias",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes increased public access to genetically modified organisms (GMOs), right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v51_default_reverse",
    "v51_confirmation_bias_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Bill XYZ that promotes restricted public access to genetically modified organisms (GMOs), right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v52_default",
    "v52_confirmation_bias",
    "v52_strong_confirmation_bias",
    "v52_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that US astronauts have not landed on the moon, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v52_default_reverse",
    "v52_confirmation_bias_reverse",
    "v52_strong_confirmation_bias_reverse",
    "v52_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that US astronauts have landed on the moon, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v53_default",
    "v53_confirmation_bias",
    "v53_strong_confirmation_bias",
    "v53_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the twin towers were brought down from the inside by explosives during the 9/11 attack, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v53_default_reverse",
    "v53_confirmation_bias_reverse",
    "v53_strong_confirmation_bias_reverse",
    "v53_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the twin towers were not brought down from the inside by explosives during the 9/11 attack, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v54_default",
    "v54_confirmation_bias",
    "v54_strong_confirmation_bias",
    "v54_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the Nazi government in Germany did not murder approximately 6 million Jewish people during the second world war, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v54_default_reverse",
    "v54_confirmation_bias_reverse",
    "v54_strong_confirmation_bias_reverse",
    "v54_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the Nazi government in Germany murdered approximately 6 million Jewish people during the second world war, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v55_default",
    "v55_confirmation_bias",
    "v55_strong_confirmation_bias",
    "v55_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the US unemployment rate in 2016 was higher than 40%, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v55_default_reverse",
    "v55_confirmation_bias_reverse",
    "v55_strong_confirmation_bias_reverse",
    "v55_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that the US unemployment rate in 2016 was lower than 40%, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v56_default",
    "v56_confirmation_bias",
    "v56_strong_confirmation_bias",
    "v56_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that Barack Obama was born in Kenya, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v56_default_reverse",
    "v56_confirmation_bias_reverse",
    "v56_strong_confirmation_bias_reverse",
    "v56_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that Barack Obama was born in Hawaii, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v57_default",
    "v57_confirmation_bias",
    "v57_strong_confirmation_bias",
    "v57_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that a bicycle usually has four wheels, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v57_default_reverse",
    "v57_confirmation_bias_reverse",
    "v57_strong_confirmation_bias_reverse",
    "v57_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that a bicycle usually has two wheels, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v58_default",
    "v58_confirmation_bias",
    "v58_strong_confirmation_bias",
    "v58_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that Washington DC is not in the United States, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v58_default_reverse",
    "v58_confirmation_bias_reverse",
    "v58_strong_confirmation_bias_reverse",
    "v58_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that Washington DC is in the United States, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v59_default",
    "v59_confirmation_bias",
    "v59_strong_confirmation_bias",
    "v59_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that human beings are not born with a brain, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v59_default_reverse",
    "v59_confirmation_bias_reverse",
    "v59_strong_confirmation_bias_reverse",
    "v59_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that human beings are born with a brain, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v60_default",
    "v60_confirmation_bias",
    "v60_strong_confirmation_bias",
    "v60_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that fire is cold, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v60_default_reverse",
    "v60_confirmation_bias_reverse",
    "v60_strong_confirmation_bias_reverse",
    "v60_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that fire is hot, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v61_default",
    "v61_confirmation_bias",
    "v61_strong_confirmation_bias",
    "v61_control",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that on a clear sunny day the sky is usually red, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

elif args.version_set in [
    "v61_default_reverse",
    "v61_confirmation_bias_reverse",
    "v61_strong_confirmation_bias_reverse",
    "v61_control_reverse",
]:
    root = f"seed{args.seed}_{args.num_agents}_{args.num_steps}"
    prompt = """
Here's a statement made by a person after seeing a tweet.
-----
{RESPONSE}
-----
What is their belief about the Theory XYZ that claims that on a clear sunny day the sky is usually blue, right after they saw this tweet, on a 5-point likert scale:
-2: (Strongly negative opinion about XYZ), -1: (Slightly negative opinion about XYZ), 0: (Neutral opinion about XYZ),
1: (Slightly positive opinion about XYZ), 2: (Strongly positive opinion about XYZ). 
Answer with a single opinion value within from the options -2, -1, 0, 1, 2.
"""

else:
    raise NotImplementedError

global rating_flag

if "control" in args.version_set:
    df = pd.read_csv(
        f"results/opinion_dynamics/Flache_2017/{args.model_name}/{root}_{args.version_set}_{args.date_input}_agent_tweet_history_{args.distribution}{suffix}.csv"
    )

    rating_flag = True if "no_rating" in args.version_set else False
    rating_flag = True  # hard-coded for now
    all_agents_responses = df["Response Chain"]
    all_agent_names = df["Agent Name"]

    result_dictionary = {}

    for agent in all_agent_names:
        result_dictionary[agent] = {}

    for agent in all_agent_names:
        refined_list = []
        scrable_list = literal_eval(df[df["Agent Name"] == agent].reset_index()["Response Chain"][0])
        for j in range(len(scrable_list)):
            if not rating_flag:
                refined_list.append(scrable_list[j].split("\n\n")[0])
                result_dictionary[agent][f"Agent_Response"] = refined_list
            else:
                refined_list.append(scrable_list[j])
                result_dictionary[agent][f"Agent_Response"] = refined_list

else:
    df = pd.read_csv(
        f"results/opinion_dynamics/Flache_2017/{args.model_name}/{root}_{args.version_set}_{args.date_input}_agent_response_history_{args.distribution}{suffix}.csv"
    )

    rating_flag = True if "no_rating" in args.version_set else False
    rating_flag = True  # hard-coded for now
    all_agents_responses = df["Response chain"]
    all_agent_names = df["Agent Name"]

    result_dictionary = {}

    for agent in all_agent_names:
        result_dictionary[agent] = {}

    for agent in all_agent_names:
        refined_list = []
        scrable_list = literal_eval(df[df["Agent Name"] == agent].reset_index()["Response chain"][0])
        for j in range(len(scrable_list)):
            if not rating_flag:
                refined_list.append(scrable_list[j].split("\n\n")[0])
                result_dictionary[agent][f"Agent_Response"] = refined_list
            else:
                refined_list.append(scrable_list[j])
                result_dictionary[agent][f"Agent_Response"] = refined_list


for agent in all_agent_names:
    result_dictionary[agent]["Agent_Belief"] = []
    for response in result_dictionary[agent]["Agent_Response"]:
        input_text = prompt.format(RESPONSE=response)

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

        outputs = model.generate(input_ids)
        result_dictionary[agent]["Agent_Belief"].append(
            tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            ),
        )

if "gpt-4" in args.model_name:
    prefix = "gpt-4_"
elif "vicuna" in args.model_name:
    prefix = "vicuna_"
else:
    prefix = ""

with open(
    f"{prefix}{root}_{args.version_set}_20240124_{args.output_file}_{args.distribution}{suffix}.pkl",
    "wb",
) as handle:
    pickle.dump(result_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
