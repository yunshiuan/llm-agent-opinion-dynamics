"""
- control_v5
  - disallow the agents from interacting with each other
  - remove persona altogether
  - each model provides 10 responses about the topic (without tweeting or seeing tweets)
  - for t in 1:10:
    - report its belief about the topic
"""
import os
from os.path import join
from collections import defaultdict
import pandas as pd
import re
import csv
from datetime import date
import argparse
from numpy.random import choice, shuffle

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#######################
# Build argument parser
#######################
parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Script")
parser.add_argument(
    "-m",
    "--model_name",
    default="gpt-3.5-turbo-16k",
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


if os.getenv("OPENAI_API_KEY"):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
else:
    with open("openai-key.txt", "r") as f:
        OPENAI_API_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class Agent:
    def __init__(
        self, agent_id, persona, model_name, temperature, max_tokens, prompt_template_root
    ):
        self.agent_id = agent_id
        self.prompt_template_root = prompt_template_root
        if persona is not None:
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
            persona_prompt = HumanMessagePromptTemplate.from_template(self.persona)

        self.count_opinion_reported = 0
        self.previous_interaction_type = "none"

        llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # Create placeholders for back-and-forth history with the LLM
        history_message_holder = MessagesPlaceholder(variable_name="history")
        quesion_placeholder = HumanMessagePromptTemplate.from_template("{input}")

        if persona is not None:
            with open(
                join(self.prompt_template_root, args.version_set, "step1_persona.md"), "r"
            ) as f:
                sys_prompt = f.read()
            sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
                AGENT_PERSONA=self.persona, AGENT_NAME=self.agent_name
            )
            systems_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)

        # Initialize the LLM agent with the language chain and its memory
        chat_prompt = ChatPromptTemplate.from_messages(
            [history_message_holder, quesion_placeholder]
        )
        memory = ConversationBufferMemory(return_messages=True, human_prefix="Game Master")  # Add
        agent_conversation = ConversationChain(
            llm=llm, memory=memory, prompt=chat_prompt, verbose=True
        )
        self.memory = agent_conversation

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
        if previous_interaction_type == "none":
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step1_report_opinion_prev_none.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0]
            response = get_integer_llm_response(self.memory, prompt)

        elif previous_interaction_type == "report":
            with open(
                join(
                    self.prompt_template_root,
                    args.version_set,
                    "step1_report_opinion_prev_report.md",
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0]
            response = get_integer_llm_response(self.memory, prompt)

        if not add_to_memory:
            self.memory.memory.chat_memory.messages.pop()
            self.memory.memory.chat_memory.messages.pop()

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
            with open(
                join(
                    self.prompt_template_root,
                    args.version_set,
                    "step1b_add_to_memory_prev_none_cur_report.md",
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                REPORT_COUNT=report_count,
                SUPERSCRIPT=get_superscript(report_count),
                OPINION_REPORTED=opinion_reported,
            )
            self.memory.memory.chat_memory.add_user_message(prompt)

        elif previos_interaction_type == "report" and current_interaction_type == "report":
            with open(
                join(
                    self.prompt_template_root,
                    args.version_set,
                    "step1b_add_to_memory_prev_report_cur_report.md",
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                REPORT_COUNT=report_count,
                SUPERSCRIPT=get_superscript(report_count),
                OPINION_REPORTED=opinion_reported,
            )
            self.memory.memory.chat_memory.add_user_message(prompt)

    def get_count_opinion_reported(self):
        """Get the number of opinions that the agent has reported so far.

        Returns:
            int: the number of opinions that the agent has reported so far.
        """
        return self.count_opinion_reported

    def increase_count_opinion_reported(self):
        """Increase the number of opinions that the agent has reported so far by 1."""
        self.count_opinion_reported += 1


def get_superscript(count):
    if count in [1, 21]:
        return "st"
    elif count in [2, 22]:
        return "nd"
    elif count in [3, 23]:
        return "rd"
    elif count in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        return "th"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def get_llm_response(conversation, prompt):
    return conversation.predict(input=prompt)


def get_integer_llm_response(conversation, prompt):
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        response = get_llm_response(conversation, prompt)
        try:
            if len(re.findall("[+-]?([0-9]*)?[.][0-9]+", response)) > 0:
                print("Bad (non-integral) value found, re-prompting agent...")
                raise ValueError
            break
        except ValueError:
            attempts += 1
            if attempts == 3:
                import pdb

                pdb.set_trace()
            conversation.memory.chat_memory.messages.pop()

    if attempts == max_attempts:
        print("Failed to get a valid integer Likert scale belief after 3 attempts.")
        raise ValueError

    return response


def main(
    num_steps,
    model_name,
    temperature,
    max_tokens,
    prompt_template_root,
    path_result,
    date_version,
):
    # ------------------
    # only one single agent is used
    # - since there is no persona, the agent is simply the LLM itself
    # ------------------
    agent = Agent(
        agent_id=None,
        persona=None,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_template_root=prompt_template_root,
    )

    # ------------------
    # results collectors
    # ------------------
    # - dict_agent_report {agent_id: [(t1,report_1),(t6,report_6),(t12,report_12)]}
    dict_agent_report = defaultdict(list)
    list_agents = [agent]

    for t in range(num_steps):
        # ------------------
        # agent j reports its opinion
        # - ask for j's opinion
        # - the prompt is different depending on agent j's interaction history: 1) no history, 2) previous interaction = report opinion
        # ------------------
        agent_previos_interaction_type = agent.previous_interaction_type
        if agent_previos_interaction_type in ["none", "report"]:
            report_j = agent.report_opinion(
                previous_interaction_type=agent.previous_interaction_type,
                report_count=agent.get_count_opinion_reported(),
                add_to_memory=False,
            )
            agent.increase_count_opinion_reported()
        else:
            raise ValueError(
                "agent_j_previos_interaction_type is not valid: {}".format(
                    agent_previos_interaction_type
                )
            )

        agent.add_to_memory(
            opinion_reported=report_j,
            previos_interaction_type=agent_previos_interaction_type,
            current_interaction_type="report",
            report_count=agent.get_count_opinion_reported(),
        )

        agent.previous_interaction_type = "report"
        dict_agent_report[agent].append((t + 1, report_j))

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
        + "_agent_response_history_"
        + args.distribution
        + ".csv"
    )
    with open(out_name, "w+") as g:
        writer = csv.writer(g, delimiter=",")
        writer.writerow(
            [
                "Time Step",
                "Response Chain",
            ]
        )

        for agent in dict_agent_report.keys():
            time_step_changes = [time_step[0] for time_step in dict_agent_report[agent]]
            opinion_chain = [time_step[1] for time_step in dict_agent_report[agent]]
            row = []
            row.append(list(time_step_changes))
            row.append(list(opinion_chain))

            writer.writerow(row)

    return


if __name__ == "__main__":
    experiment_id = "Flache_2017"
    model_name = args.model_name
    temperature = args.temperature
    max_tokens = 300
    num_steps = args.num_steps
    prompt_template_root = "prompts/opinion_dynamics/Flache_2017"
    path_result = "results/opinion_dynamics/{}/{}".format(experiment_id, model_name)
    if args.test_run:
        path_result = "results/opinion_dynamics/{}/{}/test_runs".format(experiment_id, model_name)
    today = date.today()
    today = today.strftime("%Y%m%d")
    date_version = today

    global rating_flag
    rating_flag = True if "no_rating" in args.version_set else False
    rating_flag = args.no_rating

    main(
        num_steps=num_steps,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_template_root=prompt_template_root,
        date_version=date_version,
        path_result=path_result,
    )
