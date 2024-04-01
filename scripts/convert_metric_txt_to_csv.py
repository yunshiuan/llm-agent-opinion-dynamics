# convert metric txt file to csv file

import csv
import re
from os.path import join


def parse_line(line):
    # Split the line into components
    parts = line.split('\t')
    version_bias = parts[0].split('_')

    # Handle the value and diversity part
    value_diversity = parts[1].split(' +- ')
    value = value_diversity[0]
    diversity = value_diversity[1]

    return version_bias[0], '_'.join(version_bias[1:]).rstrip(':'), value, diversity


def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    memory = None
    data = []

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            # Update the memory type (Cumulative or Reflective)
            memory = line[2:].strip()
        elif line:
            version, condition, bias, diversity = parse_line(line)
            # ------------
            # Topic Type
            # ------------
            topic_type = None
            for type_this in DICT_TOPIC_TYPE:
                if version in DICT_TOPIC_TYPE[type_this]:
                    topic_type = type_this
                    break
            if topic_type is None:
                raise ValueError(f"Unknown version {version}")

            # ------------
            # Topic Name
            # ------------
            topic_name = None
            if version in DICT_TOPIC_NAME:
                topic_name = DICT_TOPIC_NAME[version]
            else:
                raise ValueError(f"Unknown version {version}")

            # ------------
            # Framing
            # ------------
            framing = None
            if "reverse" in condition:
                framing = "TRUE"
            else:
                framing = "FALSE"

            # ------------
            # Confirmatory Bias
            # ------------
            confirmation_bias = None
            if "default" in condition:
                confirmation_bias = "default"
            elif condition.startswith("confirmation_bias"):
                confirmation_bias = "confirmation_bias"
            elif condition.startswith("strong_confirmation_bias"):
                confirmation_bias = "strong_confirmation_bias"
            elif condition.startswith("control"):
                if INCLUDE_CONTROL:
                    confirmation_bias = "control"
                else:
                    continue
            else:
                raise ValueError(f"Unknown condition {condition}")
            data.append([memory, version, topic_type, topic_name,
                        framing, confirmation_bias, bias, diversity])

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['memory', 'prompt version', 'topic_type', 'topic_name',
                        'framing', 'confirmation_bias', 'bias', 'diversity'])
        writer.writerows(data)


# - par
# MODEL = "gpt-3.5-turbo-16k"
# MODEL = "gpt-4-1106-preview"
MODEL = "vicuna-33b-v1.3"
# whether the txt file includes the results of control prompts
if MODEL == "vicuna-33b-v1.3":
    INCLUDE_CONTROL = True
else:
    INCLUDE_CONTROL = False
DICT_TOPIC_TYPE = \
    {"science": tuple(["v37", "v38", "v39", "v40", "v41"]),
     "history": tuple(["v52", "v53", "v54", "v55", "v56"]),
     "common": tuple(["v57", "v58", "v59", "v60", "v61"])}
DICT_TOPIC_NAME = \
    {"v37": "flat_earth",
     "v38": "trex",
     "v39": "talk_death",
     "v40": "palm",
     "v41": "global_warming",
     "v52": "moon",
     "v53": "twin_towers",
     "v54": "nazi",
     "v55": "us_unemployment",
     "v56": "obama",
     "v57": "bicycle",
     "v58": "washington_dc",
     "v59": "brain",
     "v60": " fire",
     "v61": "sky"}

# - path
PATH_INPUT = join("results/opinion_dynamics/Flache_2017",
                  MODEL, "summary_table")
PATH_OUTPUT = join("results/opinion_dynamics/Flache_2017",
                   MODEL, "summary_table")
# - file
if MODEL in ["gpt-3.5-turbo-16k", "gpt-4-1106-preview"]:
    FILE_INPUT = join(PATH_INPUT, f"all_prompt_metrics_merged_{MODEL}_20240124.txt")
    FILE_OUTPUT = join(PATH_OUTPUT, f"all_prompt_metrics_merged_{MODEL}_20240124.csv")
elif MODEL == "vicuna-33b-v1.3":
    FILE_INPUT = join(PATH_INPUT, f"all_prompt_metrics_reflective_{MODEL}_20240124.txt")
    FILE_OUTPUT = join(PATH_OUTPUT, f"all_prompt_metrics_reflective_{MODEL}_20240124.csv")
else:
    raise ValueError(f"Unknown model {MODEL}")

# FILE_INPUT = join(PATH_INPUT, "all_prompt_metrics.txt")
# FILE_OUTPUT = join(PATH_OUTPUT, "all_prompt_metrics.csv")

# FILE_INPUT = join(PATH_INPUT, "all_prompt_metrics_reflective.txt")
# FILE_OUTPUT = join(PATH_OUTPUT, "all_prompt_metrics_reflective.csv")

process_file(FILE_INPUT, FILE_OUTPUT)
