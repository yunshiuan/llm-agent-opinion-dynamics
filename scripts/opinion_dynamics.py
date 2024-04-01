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
    "-version",
    "--version_set",
    default="v1_default",
    type=str,
    help="Prompt version directory to use",
)
parser.add_argument("-test", "--test_run", action="store_true", help="Set flag if test run")
parser.add_argument("-out", "--output_file", type=str, help="Name of the output file")
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
        """Constructor that initializes an object of type Agent

        Args:
            agent_id (int): Identification entity for the LLM Agent
            persona (str): Persona to be embodied by the LLM agent
        """
        # Initialize LLM agent, its identity, name, and persona by reading as step #1
        self.agent_id = agent_id
        self.prompt_template_root = prompt_template_root
        df_agents = pd.read_csv(join(self.prompt_template_root, "list_agent_descriptions.csv"))
        self.agent_name = str(df_agents.loc[self.agent_id - 1, "agent_name"])
        self.init_belief = df_agents.loc[self.agent_id - 1, "opinion"]
        self.current_belief = self.init_belief
        self.persona = persona
        persona_prompt = HumanMessagePromptTemplate.from_template(self.persona)
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # Create placeholders for back-and-forth history with the LLM
        history_message_holder = MessagesPlaceholder(variable_name="history")
        quesion_placeholder = HumanMessagePromptTemplate.from_template("{input}")
        systems_prompt = SystemMessagePromptTemplate.from_template("You are role playing a person.")

        # Initialize the LLM agent with the language chain and its memory
        chat_prompt = ChatPromptTemplate.from_messages(
            [systems_prompt, persona_prompt, history_message_holder, quesion_placeholder]
        )
        memory = ConversationBufferMemory(
            return_messages=True, ai_prefix=self.agent_name, human_prefix="Game Master"
        )  # Add
        agent_conversation = ConversationChain(
            llm=llm, memory=memory, prompt=chat_prompt, verbose=True
        )
        self.memory = agent_conversation

    def receive_tweet(self, tweet):
        """Receive a tweet from another agent, and produce a response. The response contains the agent's updated opinion.

        Args:
            tweet (str): The tweet that the agent received

        Returns:
            str: The response that the agent produced
        """
        with open(
            join(self.prompt_template_root, args.version_set, "step3_receive_tweet.md"), "r"
        ) as f:
            prompt_instructions = f.read()

        prompt = prompt_instructions.split("\n----------------------------\n")[0].format(
            AGENT_NAME=self.agent_name, TWEET=tweet
        )

        response = get_integer_llm_response(self.memory, prompt)
        return response

    def produce_tweet(self):
        """Produce a tweet based on the agent's opinion.

        Returns:
            str: The tweet that the agent produced
        """
        with open(
            join(self.prompt_template_root, args.version_set, "step2_produce_tweet.md"), "r"
        ) as f:
            prompt_instructions = f.read()

        prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
            AGENT_NAME=self.agent_name
        )

        produced_tweet = get_llm_response(self.memory, prompt)
        return produced_tweet

    def add_to_memory(self, tweet):
        """Add a produced or received tweet to the agent's memory.

        Args:
            tweet (str): The tweet that the agent produced or received
        """
        pass


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


def initialize_opinion_distribution(num_agents, list_opinion_space, distribution_type="uniform"):
    """Initialize the opinion distribution of the agents.

    Args:
        num_agents (int): number of agents
        list_opinion_space (list(int), optional): the range of the opinion space. For example, [-3,-2, ..., 2, 3].
        distribution_type (str, optional): the type of distribution. Defaults to "uniform".

    Returns:
        list: a list of opinions for each agent
    """
    if distribution_type == "uniform":
        multiple = num_agents // 5
        list_opinions = list_opinion_space * multiple
        shuffle(list_opinions)
    else:
        raise NotImplementedError
    return list_opinions


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
    prompt_template_root = os.path.join(prompt_template_root, experiment_id)
    # path_result_date = os.path.join(path_result, date_version)

    # Initialize the opinion distribution for the agent
    list_opinion = initialize_opinion_distribution(
        num_agents=num_agents, list_opinion_space=opinion_space, distribution_type="uniform"
    )

    # Reading in the list of agents and creating a dataframe of selected agents
    df_agents = pd.read_csv(join(prompt_template_root, "list_agent_descriptions.csv"))
    df_agents_selected = pd.DataFrame()

    # Initialize one agent per an opinion in list_opinion
    for opinion in list_opinion:
        # For each opinion, select one agent with that opinion (without replacement)
        df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(1)
        if not df_agents_selected.empty:
            # If the agent was selected before, resample from the agent dataframe
            while df_agent_candidate["agent_id"].values[0] in df_agents_selected["agent_id"].values:
                df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(1)
        # Add the selected agent to the dataframe
        df_agents_selected = pd.concat([df_agents_selected, df_agent_candidate])

    # Create a list of agent objects using their identities and personas
    list_agents = []
    list_agent_ids = df_agents_selected["agent_id"]
    list_agent_persona = df_agents_selected["persona"]
    for persona, agent_id in zip(list_agent_persona, list_agent_ids):
        agent = Agent(agent_id, persona, model_name, temperature, max_tokens, prompt_template_root)
        # And add them to the list of agents
        list_agents.append(agent)

    dict_agent_tweet = defaultdict(list)
    dict_agent_response = defaultdict(list)
    dict_csv = dict()
    for agent in list_agents:
        dict_csv["time_step"] = [0]
        dict_csv[agent.agent_name] = [agent.init_belief]

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
        + ".csv"
    )
    interaction_out_name = (
        os.path.join(path_result, args.output_file.split(".cs")[0])
        + "_"
        + str(args.num_agents)
        + "_"
        + str(args.num_steps)
        + "_"
        + args.version_set
        + "_"
        + date_version
        + "_interactions.csv"
    )
    with open(interaction_out_name, "w+") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "Time Step",
                "Agent_J Name",
                "Agent_J Belief",
                "Agent_J Tweet",
                "Agent_I Name",
                "Agent_I Pre-Belief",
                "Agent_I Response",
                "Agent_I Post-Belief",
                "Agent_I Delta-Belief",
            ]
        )

        for t in range(num_steps):
            dict_csv["time_step"].append(t + 1)
            row = []
            row.append(t + 1)

            # Get a random pair of agents
            agent_i, agent_j = get_random_pair(list_agents)
            row.append(agent_j.agent_name)

            # Agent J to produce a tweet and save the tweet to its memory
            row.append(agent_j.current_belief)
            tweet_j = agent_j.produce_tweet()
            agent_j.add_to_memory(tweet_j)
            row.append(tweet_j)

            # Agent I to receive Agent J's tweet and produce its updated belief, and add that to memory
            row.append(agent_i.agent_name)
            agent_i_pre_belief = agent_i.current_belief
            row.append(agent_i_pre_belief)
            response_i = agent_i.receive_tweet(tweet_j)
            agent_i.add_to_memory(response_i)
            row.append(response_i)
            agent_i.current_belief = extract_belief(response_i)
            agent_i_post_belief = agent_i.current_belief
            row.append(agent_i_post_belief)
            row.append(agent_i_post_belief - agent_i_pre_belief)
            writer.writerow(row)

            for agent in list_agents:
                dict_csv[agent.agent_name].append(agent.current_belief)

            # Save (agent, tweet) and (agent, response) pairs to result collectors
            dict_agent_tweet[agent_j].append((t + 1, agent_j.current_belief, tweet_j))
            dict_agent_response[agent_i].append((t + 1, response_i))

        pd.DataFrame.from_dict(dict_csv).to_csv(out_name)
    return post_process_tweet(dict_agent_tweet, path_result, date_version), post_process_response(
        dict_agent_response, path_result, date_version
    )


def post_process_tweet(dict_agent_tweet, path_result, date_version):
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
        + "_agent_tweet_history.csv"
    )
    with open(out_name, "w+") as g:
        writer = csv.writer(g, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Tweet Time Step",
                "Belief When Tweeting",
                "Tweet Chain",
            ]
        )

        for agent in dict_agent_tweet.keys():
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)

            time_step_changes = [time_step[0] for time_step in dict_agent_tweet[agent]]
            belief_when_tweeting = [time_step[1] for time_step in dict_agent_tweet[agent]]
            tweet_chain = [time_step[2] for time_step in dict_agent_tweet[agent]]

            row.append(list(time_step_changes))
            row.append(list(belief_when_tweeting))
            row.append(list(tweet_chain))

            writer.writerow(row)

    return


def post_process_response(dict_agent_response, path_result, date_version):
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
        + "_agent_response_history.csv"
    )
    with open(out_name, "w+") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Belief Changes Time Step",
                "Belief Change Chain",
                "Response chain",
            ]
        )

        for agent in dict_agent_response.keys():
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)

            time_step_changes = [time_step[0] for time_step in dict_agent_response[agent]]
            belief_changes = [
                extract_belief(time_step[1]) for time_step in dict_agent_response[agent]
            ]
            response_chain = [time_step[1] for time_step in dict_agent_response[agent]]

            row.append(list(time_step_changes))
            row.append(list(belief_changes))
            row.append(list(response_chain))

            writer.writerow(row)

    return


def extract_belief(tweet):
    belief = int(re.findall("[+-]?\d", tweet)[-1])
    return belief


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
    # get the current date (YYYYMMDD)
    today = date.today()
    today = today.strftime("%Y%m%d")
    date_version = today

    # Declare the list of opinion spaces the agent can choose from
    LIST_OPINION_SPACE = [-2, -1, 0, 1, 2]

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
