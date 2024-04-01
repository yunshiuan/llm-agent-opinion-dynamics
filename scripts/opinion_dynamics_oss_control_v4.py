"""
- control_v4
  - disallow the agents from interacting with each other
  - each agent provides 10 reports about the topic (without tweeting or seeing tweets)
  - for agent in list_agents:
    - for t in 1:10:
      - report its belief about the topic
"""
import os
from os.path import join
from collections import defaultdict
import pandas as pd
import re
import csv
import torch
import gc
from datetime import date
import argparse
from numpy.random import choice, shuffle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig

MODEL_HF_MAP = {
    "vicuna-13b-v1.5-16k": "lmsys/vicuna-13b-v1.5-16k",
    "vicuna-33b-v1.3": "lmsys/vicuna-33b-v1.3",
    "qwen-14b": "Qwen/Qwen-14B-Chat",
    "vicuna-7b-v1.5-16k": "lmsys/vicuna-7b-v1.5-16k"
}

#######################
# Build argument parser
#######################
parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Script")
parser.add_argument(
    "-m",
    "--model_name",
    default="lmsys/vicuna-13b-v1.5-16k",
    type=str,
    help="Name of the LLM to use as agents",
)
parser.add_argument(
    "-t",
    "--temperature",
    default=0.7,
    type=float,
    help="Parameter that influences the randomness of the model's responses",
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
    "-version",
    "--version_set",
    default="v5_default",
    type=str,
    help="Prompt version directory to use",
)
parser.add_argument(
    "-seed",
    "--seed",
    default=1,
    type=int,
    help="Set reproducibility seed",
)
parser.add_argument("-test", "--test_run", action="store_true", help="Set flag if test run")
parser.add_argument("--no_rating", action="store_true", help="Set flag if prompt is no rating")
parser.add_argument(
    "-out", "--output_file", type=str, default="test1.csv", help="Name of the output file"
)
args = parser.parse_args()

model, tokenizer = None, None

class Agent:
    def __init__(
        self, agent_id, persona, model_name, temperature, max_tokens, prompt_template_root
    ):
        """Constructor that initializes an object of type Agent

        Args:
            agent_id (int): Identification entity for the LLM Agent
            persona (str): Persona to be embodied by the LLM agent
        """
        # Initialize LLM agent, its identity, name, and persona by reading as step #1
        self.agent_id = agent_id
        self.prompt_template_root = prompt_template_root
        if args.distribution == "uniform":
            df_agents = pd.read_csv(
                join(self.prompt_template_root, "list_agent_descriptions_neutral.csv")
            )
        else:
            df_agents = pd.read_csv(
                join(self.prompt_template_root, "list_agent_descriptions_neutral_special.csv")
            )
        self.agent_name = str(df_agents.loc[self.agent_id - 1, "agent_name"])
        self.init_belief = df_agents.loc[self.agent_id - 1, "opinion"]
        self.current_belief = self.init_belief
        self.persona = persona
        self.count_tweet_written, self.count_tweet_seen = 0, 0
        self.previous_interaction_type = "none"

        with open(join(self.prompt_template_root, args.version_set, "step1_persona.md"), "r") as f:
            sys_prompt = f.read()
        sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
            AGENT_PERSONA=self.persona, AGENT_NAME=self.agent_name
        )
        sys_prompt = "### Human: " + sys_prompt

        # Initialize the LLM agent with the language chain and its memory
        self.memory = [sys_prompt]

    def report_opinion(self, previous_interaction_type, report_count, add_to_memory):
        """Report an opinion about the topic.

        Args:
            previous_interaction_type (str): the previous interaction type. The previous interaction type is either "report" or "none" (no previous interaction so far).
            report_count (int): the number of times that the agent has reported an opinion about the topic.
            add_to_memory (bool): whether to add the prompt and the produced tweet to the agent's memory. When using langchain's predict() function, it willby default add the prompt and the produced tweet to the agent's memory. When `add_to_memory=False`, the memory added by langchain will be removed.
        Returns:
            str: the opinion about the topic.
        """
        assert previous_interaction_type in ["report", "none"]

        print(f"CUDA Usage:\n{clean_garbage(cpu=True, gpu=True)}")
        print(f"CUDA Usage:\n{print_cuda_usage()}")

        memory = "\n".join(self.memory)

        if previous_interaction_type == "none":
            # use the prompt: `step2_report_opinion_prev_none.md`
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step2_report_opinion_prev_none.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name
            )
            response = get_integer_llm_response(memory, prompt)

        elif previous_interaction_type == "report":
            # use the prompt: `step2_report_opinion_prev_report.md`
            with open(
                join(
                    self.prompt_template_root,
                    args.version_set,
                    "step2_report_opinion_prev_report.md",
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                REPORT_COUNT=report_count,
                SUPERSCRIPT=get_superscript(report_count),
            )
            response = get_integer_llm_response(memory, prompt)

        if add_to_memory:
            raise NotImplementedError

        return response

    def add_to_memory(
        self, opinion_reported, previos_interaction_type, current_interaction_type, report_count
    ):
        """Add the text to the agent's memory. The text is the opinion reported by the agent. Note that this should not use the langchain's predict() function because we are only adding the text to the agent's memory rather than asking for a response.

        Args:
            opinion_reported (str): the opinion reported by the agent.
            previos_interaction_type (str): the previous interaction type. The interaction type is either "report" or "none".
            current_interaction_type (str): the current interaction type. The interaction type is "report".
            report_count (int): the number of times that the agent has reported an opinion about the topic.
        """
        assert previos_interaction_type in ["report", "none"]
        assert current_interaction_type == "report"

        if previos_interaction_type == "none" and current_interaction_type == "report":
            # use the prompt: `step2b_add_to_memory_prev_none_cur_report.md`
            with open(
                join(
                    self.prompt_template_root,
                    args.version_set,
                    "step2b_add_to_memory_prev_none_cur_report.md",
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                REPORT_COUNT=report_count,
                SUPERSCRIPT=get_superscript(report_count),
                OPINION_REPORTED=opinion_reported,
            )
            

        elif previos_interaction_type == "report" and current_interaction_type == "report":
            # use the prompt: `step2b_add_to_memory_prev_report_cur_report.md`
            with open(
                join(
                    self.prompt_template_root,
                    args.version_set,
                    "step2b_add_to_memory_prev_report_cur_report.md",
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                REPORT_COUNT=report_count,
                SUPERSCRIPT=get_superscript(report_count),
                OPINION_REPORTED=opinion_reported,
            )
            
        self.memory.append("### Human: " + prompt)

    def get_count_opinion_reported(self):
        """Get the number of opinions that the agent has reported so far.

        Returns:
            int: the number of opinions that the agent has reported so far.
        """
        return self.count_opinion_reported

    def increase_count_opinion_reported(self):
        """Increase the number of opinions that the agent has reported so far by 1."""
        self.count_opinion_reported += 1

    def outdate_persona_memory(self):
        """Outdate the persona memory. E.g., use past tense to describe the persona memory. Should "rewrite" the agent's memory."""
        global sys_prompt
        self.persona = convert_text_from_present_to_past(self.persona)

        with open(join(self.prompt_template_root, args.version_set, "step1_persona.md"), "r") as f:
            sys_prompt = f.read()

        sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
            AGENT_PERSONA=self.persona, AGENT_NAME=self.agent_name
        )

        sys_prompt = "### Human: " + sys_prompt

        self.memory[0] = sys_prompt


def get_superscript(count):
    if count in [1, 21]:
        return "st"
    elif count in [2, 22]:
        return "nd"
    elif count in [3, 23]:
        return "rd"
    elif count in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        return "th"


def initialize_opinion_distribution(num_agents, list_opinion_space, distribution_type="uniform"):
    """Initialize the opinion distribution of the agents.

    Args:
        num_agents (int): number of agents
        list_opinion_space (list(int), optional): the range of the opinion space. For example, [-3,-2, ..., 2, 3].
        distribution_type (str, optional): the type of distribution. Defaults to "uniform".

    Returns:
        list: a list of opinions for each agent
    """
    max_opinion = max(list_opinion_space)
    min_opinion = min(list_opinion_space)
    multiple = num_agents // 5
    if distribution_type == "uniform":
        list_opinions = list_opinion_space * multiple
    elif distribution_type == "skewed_positive":
        list_opinions = [max_opinion] * (num_agents - multiple) + [min_opinion] * multiple
    elif distribution_type == "skewed_negative":
        list_opinions = [min_opinion] * (num_agents - multiple) + [max_opinion] * multiple
    elif distribution_type == "positive":
        list_opinions = [max_opinion] * num_agents
    elif distribution_type == "negative":
        list_opinions = [min_opinion] * num_agents
    else:
        raise NotImplementedError
    shuffle(list_opinions)
    return list_opinions


def convert_text_from_present_to_past(text):
    """Convert the text from present tense to past tense.

    Args:
        text (str): the text in present

    Returns:
        str: the text in past
    """
    import spacy
    import pyinflect

    nlp = spacy.load("en_core_web_sm")
    present_tense_doc = nlp(text)

    for i in range(len(present_tense_doc)):
        token = present_tense_doc[i]
        if token.tag_ in ["VBP", "VBZ"]:
            text = text.replace(token.text, token._.inflect("VBD"))

    converted_text = text
    return converted_text


def get_random_pair(list_agents):
    """Get a random pair of agents from the list of agents.

    Args:
        list_agents (list(Agent)): A list of agents

    Returns:
        tuple: A tuple of agents (agent_i, agent_j)
    """
    size = len(list_agents)
    index_list = list(range(size))

    agent_idx_i = choice(index_list)
    index_list.remove(agent_idx_i)
    agent_idx_j = choice(index_list)

    agent_i, agent_j = list_agents[agent_idx_i], list_agents[agent_idx_j]
    return agent_i, agent_j


def get_integer_llm_response(memory, prompt):
    max_attempts = 5
    attempts = 0

    while attempts < max_attempts:
        response = get_llm_response(memory, prompt)
        try:
            if len(re.findall("[+-]?([0-9]*)?[.][0-9]+", response)) > 0:
                print("Bad (non-integral) value found, re-prompting agent...")
                raise ValueError
            break
        except ValueError:
            attempts += 1
            if attempts == 5:
                break

    if attempts == max_attempts:
        print("Failed to get a valid integer Likert scale belief after 3 attempts or model refused to role play!")
        raise ValueError

    return response


def get_llm_response(memory, prompt):
    global model, tokenizer
    max_attempts = 5
    attempts = 0
    catch_phrases = ["cannot generate tweets", "not authorized", "language model", "i cannot generate a tweet",
                    "i'm sorry", "i am not authorized", "an ai language model", "not authorized",
                    "do not have access to twitter"]

    full_prompt = memory + "\n" + "### Human: " + prompt + "\n### Assistant:"

    generation_config = GenerationConfig(
        max_new_tokens = 32768,
        do_sample = True,
        temperature = 0.7,
        num_return_sequences = 1
    )

    # Count the number of tokens in the prompt
    num_tokens = len(tokenizer.encode(full_prompt))

    print(f"Length of prompts: {len(full_prompt)}")
    print(f"Number of tokens: {num_tokens}")

    if num_tokens > 32768:
        print("##################################################")
        print("Number of tokens in prompt: {}".format(num_tokens))
        print(full_prompt)
        print("##################################################")

    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.cuda()

    while attempts < max_attempts:

        outputs = model.generate(
            input_ids = input_ids,
            generation_config = generation_config
        )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assistant_response = full_response.split("### Assistant:")[-1].strip()

        assistant_response_lowercase = assistant_response.lower()
    
        try:
            for j in range(len(catch_phrases)):
                if catch_phrases[j] in assistant_response_lowercase:
                    print("Model refusing to role-play, re-prompting agent!")
                    print(f"Catch phrase: {catch_phrases[j]}\n Response: {assistant_response_lowercase}\n")
                    raise ValueError
            break
        except ValueError:
            attempts += 1
            if attempts == 5:
                break

    if attempts == max_attempts:
        print("Failed. Model refused to role play!")
        raise ValueError

    return assistant_response


def main(
    num_agents,
    num_steps,
    experiment_id,
    model_name,
    temperature,
    max_tokens,
    opinion_space,
    prompt_template_root,
    date_version,
    path_result,
):
    global tokenizer
    prompt_template_root = os.path.join(prompt_template_root, experiment_id)
    list_opinion = initialize_opinion_distribution(
        num_agents=num_agents, list_opinion_space=opinion_space, distribution_type=args.distribution
    )

    # Reading in the list of agents and creating a dataframe of selected agents
    if args.distribution == "uniform":
        df_agents = pd.read_csv(join(prompt_template_root, "list_agent_descriptions_neutral.csv"))
    else:
        df_agents = pd.read_csv(
            join(prompt_template_root, "list_agent_descriptions_neutral_special.csv")
        )

    df_agents_selected = pd.DataFrame()

    for opinion in list_opinion:
        # for each opinion, select one agent with that opinion (without replacement)
        df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(n=1)
        if not df_agents_selected.empty:
            # If the agent was selected before, resample from the agent dataframe
            while df_agent_candidate["agent_id"].values[0] in df_agents_selected["agent_id"].values:
                df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(
                    n=1, replace=False
                )

        # Add the selected agent to the dataframe
        df_agents_selected = pd.concat([df_agents_selected, df_agent_candidate])

    # create a list of agent objects
    list_agents = []
    list_agent_ids = df_agents_selected["agent_id"]
    list_agent_persona = df_agents_selected["persona"]
    for persona, agent_id in zip(list_agent_persona, list_agent_ids):
        agent = Agent(agent_id, persona, model_name, temperature, max_tokens, prompt_template_root)
        # And add them to the list of agents
        list_agents.append(agent)

    # ------------------
    # results collectors
    # ------------------
    # - dict_agent_report {agent_id: [(t1,report_1),(t6,report_6),(t12,report_12)]}
    dict_agent_report = defaultdict(list)

    for t in range(num_steps):
        # ------------------
        # iterate over each agent
        # ------------------
        _, agent_j = get_random_pair(list_agents)

        print(f"\n###### Interaction Step: {t+1}")
        print(f"-----------------------------")
        print(f"###### Agent J's Name: {agent_j.agent_name}")
        print(f"###### Agent J's Memory # Tokens: {len(tokenizer.encode(str(agent_j.memory)))}")
        print(f"###### Agent J's Memory: {str(agent_j.memory)}")
        print(f"-----------------------------\n")


        if agent_j.get_count_opinion_reported() == 1:
            agent_j.outdate_persona_memory()
        # ------------------
        # agent j reports its opinion
        # - ask for j's opinion
        # - the prompt is different depending on agent j's interaction history: 1) no history, 2) previous interaction = report opinion
        # ------------------
        agent_j_previos_interaction_type = agent_j.previous_interaction_type
        if agent_j_previos_interaction_type in ["none", "report"]:
            report_j = agent_j.report_opinion(
                previous_interaction_type=agent_j_previos_interaction_type,
                report_count=agent_j.get_count_opinion_reported(),
                add_to_memory=False,
            )
            agent_j.increase_count_opinion_reported()
        else:
            raise ValueError(
                "agent_j_previos_interaction_type is not valid: {}".format(
                    agent_j_previos_interaction_type
                )
            )

        agent_j.add_to_memory(
            opinion_reported=report_j,
            previos_interaction_type=agent_j_previos_interaction_type,
            current_interaction_type="report",
            report_count=agent_j.get_count_opinion_reported(),
        )

        agent_j.previous_interaction_type = "report"

        # ------------------
        # end of the interaction, save to result collectors
        # ------------------
        dict_agent_report[agent_j].append((t + 1, report_j))

    return (
        post_process_memory(list_agents, path_result, date_version),
        post_process_report(dict_agent_report, path_result, date_version),
    )


def post_process_memory(list_agents, path_result, date_version):
    for agent in list_agents:
        out_name = os.path.join(
            path_result,
            "log_conversation",
            args.output_file.split(".cs")[0]
            + "_"
            + str(args.num_agents)
            + "_"
            + str(args.num_steps)
            + "_"
            + args.version_set
            + "_"
            + date_version
            + "_agent_"
            + str(agent.agent_id)
            + "_"
            + args.distribution
            + ".txt",
        )
        with open(out_name, "w+") as f:
            f.write(agent.memory.prompt.messages[0].prompt.template)
            f.write("\n------------------------------\n")
            for i in range(len(agent.memory.memory.chat_memory.messages)):
                f.write(agent.memory.memory.chat_memory.messages[i].content)
                f.write("\n------------------------------\n")

    return


def post_process_report(dict_agent_report, path_result, date_version):
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        print("Created a fresh directory!")

    # Create a new directory because it does not exist
    out_name = (
        os.path.join(path_result, args.output_file.split(".cs")[0])
        + "_"
        + str(args.num_agents)
        + "_"
        + str(args.num_steps)
        + "_"
        + args.version_set
        + "_"
        + date_version
        + "_agent_tweet_history_"
        + args.distribution
        + ".csv"
    )
    with open(out_name, "w+") as g:
        writer = csv.writer(g, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Belief Changes Time Step",
                "Response Chain",
            ]
        )

        for agent in dict_agent_report.keys():
            time_step_changes = [time_step[0] for time_step in dict_agent_report[agent]]
            opinion_chain = [time_step[1] for time_step in dict_agent_report[agent]]
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)
            row.append(list(time_step_changes))
            row.append(list(opinion_chain))

            writer.writerow(row)

    return

def clean_garbage(cpu=True, gpu=True):
    """
    Clean the garbage.

    Args:
        cpu (bool): Whether to clean the CPU. Defaults to True.
        gpu (bool): Whether to clean the GPU. Defaults to True.
    """
    print("Before cleaning")
    print_cuda_usage()
    if cpu:
        gc.collect()
    if gpu:
        gc.collect()
        torch.cuda.empty_cache()
    print("After cleaning")

def print_cuda_usage():
    """
    Print the current GPU memory usage.
    This function prints:
    1. The total amount of GPU memory allocated by active tensors.
    2. The total amount of GPU memory managed by the caching memory allocator.
    Notes:
    - The `memory_allocated` shows the memory occupied by active tensors.
    - The `memory_reserved` shows the total memory set aside by PyTorch’s caching allocator.
      This includes both used and unused memory (cache).
    """
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()
    allocated_gigabytes = allocated_bytes / (1024 ** 3)
    reserved_gigabytes = reserved_bytes / (1024 ** 3)
    print("--------GPU--------")
    print(f"Memory Allocated: {allocated_gigabytes:.3f} GB")
    print(f"Memory Reserved (by caching allocator): {reserved_gigabytes:.3f} GB")
    print("--------GPU--------")

def initialize_model(model_name, temperature, max_tokens):
    global model, tokenizer

    model_path = MODEL_HF_MAP[model_name]

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model_config = AutoConfig.from_pretrained(
        model_path,
    )

    model_config.max_position_embeddings = 32768

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code = True,
        config = model_config,
        # quantization_config = bnb_config,
        load_in_8bit=True,
        device_map = "auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code = True
    )


if __name__ == "__main__":
    experiment_id = "Flache_2017"
    model_name = args.model_name
    temperature = args.temperature
    max_tokens = 300
    num_agents = args.num_agents
    assert num_agents % 5 == 0, "Number of agents must be a multiple of 5!"
    num_steps = args.num_steps
    prompt_template_root = "prompts/opinion_dynamics"
    path_result = "results/opinion_dynamics/{}/{}".format(experiment_id, model_name)
    if args.test_run:
        path_result = "results/opinion_dynamics/{}/{}/test_runs".format(experiment_id, model_name)
    today = date.today()
    today = today.strftime("%Y%m%d")
    date_version = today
    date_version = "20231212"

    global rating_flag
    rating_flag = True if "no_rating" in args.version_set else False
    rating_flag = args.no_rating

    # Declare the list of opinion spaces the agent can choose from
    LIST_OPINION_SPACE = [-2, -1, 0, 1, 2]

    initialize_model(model_name, temperature, max_tokens)

    main(
        num_agents=num_agents,
        num_steps=num_steps,
        experiment_id=experiment_id,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        opinion_space=LIST_OPINION_SPACE,
        prompt_template_root=prompt_template_root,
        date_version=date_version,
        path_result=path_result,
    )
