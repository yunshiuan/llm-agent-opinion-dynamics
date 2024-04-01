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
    default="gpt-4-1106-preview",
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
reflection_count = 0


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

        persona_prompt = HumanMessagePromptTemplate.from_template(self.persona)
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # Create placeholders for back-and-forth history with the LLM
        history_message_holder = MessagesPlaceholder(variable_name="history")
        quesion_placeholder = HumanMessagePromptTemplate.from_template("{input}")

        with open(join(self.prompt_template_root, args.version_set, "step1_persona.md"), "r") as f:
            sys_prompt = f.read()
        sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
            AGENT_PERSONA=self.persona, AGENT_NAME=self.agent_name
        )
        systems_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)

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

    def receive_tweet(self, tweet, previous_interaction_type, tweet_written_count, add_to_memory):
        """Receive a tweet from another agent, and produce a response. The response contains the agent's updated opinion.

        Args:
            tweet (str): the tweet that the agent received.
            previous_interaction_type (str): the previous interaction type. The previous interaction type is either "tweet", "read", or "none" (no previous interaction so far).
            tweet_written_count (int): the number of tweets that the agent has written so far.
            add_to_memory (bool): whether to add the prompt and the produced tweet to the agent's memory. When using langchain's predict() function, it willby default add the prompt and the produced tweet to the agent's memory. When `add_to_memory=False`, the memory added by langchain will be removed.

        Returns:
            str: the response that the agent produced
        """
        assert previous_interaction_type in ["write", "read", "none"]

        if previous_interaction_type == "write":
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step3_receive_tweet_prev_tweet.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET_WRITTEN_COUNT=tweet_written_count,
                SUPERSCRIPT=get_superscript(tweet_written_count),
                TWEET=tweet,
            )
            response = get_integer_llm_response(self.memory, prompt)

        elif previous_interaction_type == "read":
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step3_receive_tweet_prev_read.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET=tweet,
            )
            response = get_integer_llm_response(self.memory, prompt)

        elif previous_interaction_type == "none":
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step3_receive_tweet_prev_none.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET=tweet,
            )
            response = get_integer_llm_response(self.memory, prompt)

        if not add_to_memory:
            self.memory.memory.chat_memory.messages.pop()
            self.memory.memory.chat_memory.messages.pop()

        return response

    def produce_tweet(self, previous_interaction_type, tweet_written_count, add_to_memory):
        """Produce a tweet based on the agent's opinion.

        Args:
            previous_interaction_type (str): the previous interaction type. The previous interaction type is either "tweet", "read", or "none" (no previous interaction so far).
            tweet_written_count (int): the number of tweets that the agent has written so far.
            add_to_memory (bool): whether to add the prompt and the produced tweet to the agent's memory. When using langchain's predict() function, it will by default add the prompt
                                and the produced tweet to the agent's memory. When `add_to_memory=False`, the memory added by langchain will be removed.
        Returns:
            str: the tweet that the agent produced
        """
        assert previous_interaction_type in ["write", "read", "none"]

        if previous_interaction_type == "write":
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step2_produce_tweet_prev_tweet.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET_WRITTEN_COUNT=tweet_written_count,
                SUPERSCRIPT=get_superscript(tweet_written_count),
            )
            tweet = get_llm_response(self.memory, prompt)

        elif previous_interaction_type == "read":
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step2_produce_tweet_prev_read.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
            )
            tweet = get_llm_response(self.memory, prompt)

        elif previous_interaction_type == "none":
            with open(
                join(
                    self.prompt_template_root, args.version_set, "step2_produce_tweet_prev_none.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
            )
            tweet = get_llm_response(self.memory, prompt)

        if not add_to_memory:
            self.memory.memory.chat_memory.messages.pop()
            self.memory.memory.chat_memory.messages.pop()

        return tweet

    def add_to_memory(
        self,
        previos_interaction_type,
        current_interaction_type,
        tweet_written_count,
        tweet_written=None,
        tweet_seen=None,
        response=None,
    ):
        """Add the text to the agent's memory. The text can be either a tweet written or a tweet seen + a corresponding response. Note that this should not use the langchain's predict() function because we are only adding the text to the agent's memory rather than asking for a response.

        Args:
            previos_interaction_type (str): the previous interaction type. The interaction type is either "tweet","read", or "none".
            current_interaction_type (str): the current interaction type. The interaction type is either "tweet" or "read".
            tweet_written_count (int): the number of tweets that the agent has written so far.
            tweet_written (str): the tweet the agent saw. To be added to the agent's memory. Defaults to None. Required when `current_interaction_type="tweet"`.
            tweet_seen (str): the tweet the agent saw. To be added to the agent's memory. Defaults to None. Required when `current_interaction_type="read"`.
            response (str): the response that the agent produced. To be added to the agent's memory. Defaults to None. Required when `current_interaction_type="read"`.
        """
        if current_interaction_type == "write":
            assert tweet_written is not None
        elif current_interaction_type == "read":
            assert tweet_seen is not None and response is not None
        else:
            raise ValueError(
                f"current_interaction_type must be either 'write' or 'read'. Got {current_interaction_type}"
            )
        assert previos_interaction_type in ["write", "read", "none"]
        assert current_interaction_type in ["write", "read"]

        if previos_interaction_type == "write":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        args.version_set,
                        "step2b_add_to_memory_prev_tweet_cur_tweet.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                    TWEET_WRITTEN_COUNT_LAST=tweet_written_count - 1,
                    SUPERSCRIPT_LAST=get_superscript(tweet_written_count - 1),
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self.memory.memory.chat_memory.add_user_message(prompt)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        args.version_set,
                        "step2b_add_to_memory_prev_tweet_cur_read.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_WRITTEN_COUNT=tweet_written_count,
                        SUPERSCRIPT=get_superscript(tweet_written_count),
                        TWEET_SEEN=tweet_seen,
                        REASONING=reasoning,
                        BELIEF_RATING=belief,
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_WRITTEN_COUNT=tweet_written_count,
                        SUPERSCRIPT=get_superscript(tweet_written_count),
                        TWEET_SEEN=tweet_seen,
                        REASONING=reasoning,
                    )
                self.memory.memory.chat_memory.add_user_message(prompt)

        elif previos_interaction_type == "read":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        args.version_set,
                        "step2b_add_to_memory_prev_read_cur_tweet.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self.memory.memory.chat_memory.add_user_message(prompt)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        args.version_set,
                        "step2b_add_to_memory_prev_read_cur_read.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning, BELIEF_RATING=belief
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning
                    )
                self.memory.memory.chat_memory.add_user_message(prompt)

        elif previos_interaction_type == "none":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        args.version_set,
                        "step2b_add_to_memory_prev_none_cur_tweet.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self.memory.memory.chat_memory.add_user_message(prompt)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        args.version_set,
                        "step2b_add_to_memory_prev_none_cur_read.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning, BELIEF_RATING=belief
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning
                    )
                self.memory.memory.chat_memory.add_user_message(prompt)

            global reflection_count
            reflection_count += 1
        return self.agent_reflection(reflection_count)

    def agent_reflection(self, reflection_count):
        if reflection_count == 1:
            reflection_prompt = """Now, please reflect on this experience. Summarize your experience in a few sentences."""
            agent_reflection = get_llm_response(self.memory, reflection_prompt)
        else:
            reflection_prompt = """Now, please reflect on this experience. Here is your experience so far: {}. 
            Summarize your updated experience in a few sentences.""".format(
                self.memory.memory.chat_memory.messages.pop()
            )
            agent_reflection = get_llm_response(self.memory, reflection_prompt)
        self.memory.memory.chat_memory.messages.pop()
        self.memory.memory.chat_memory.messages.pop()

        self.memory.memory.chat_memory.add_user_message(agent_reflection)

    def get_count_tweet_written(self):
        """Get the number of tweets that the agent has written so far.

        Returns:
            int: the number of tweets that the agent has written so far.
        """
        return self.count_tweet_written

    def increase_count_tweet_written(self):
        """Increase the number of tweets that the agent has written so far by 1."""
        self.count_tweet_written += 1

    def get_count_tweet_seen(self):
        """Get the number of tweets that the agent has seen so far.

        Returns:
            int: the number of tweets that the agent has seen so far.
        """
        return self.count_tweet_seen

    def increase_count_tweet_seen(self):
        """Increase the number of tweets that the agent has seen so far by 1."""
        self.count_tweet_seen += 1

    def outdate_persona_memory(self):
        """Outdate the persona memory. E.g., use past tense to describe the persona memory. Should "rewrite" the agent's memory."""
        self.persona = convert_text_from_present_to_past(self.persona)


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

    # Initialize one agent per an opinion in list_opinion
    for opinion in list_opinion:
        # For each opinion, select one agent with that opinion (without replacement)
        df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(
            n=1, random_state=args.seed
        )
        if not df_agents_selected.empty:
            # If the agent was selected before, resample from the agent dataframe
            while df_agent_candidate["agent_id"].values[0] in df_agents_selected["agent_id"].values:
                df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(
                    n=1, replace=False, random_state=args.seed + choice(range(1000))
                )
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
        + "_"
        + args.distribution
        + "_reflection.csv"
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
        + "_interactions_"
        + args.distribution
        + "_reflection.csv"
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

            # If the agent is no longer "fresh", outdate the memoery about their persona and the instruction
            if agent_i.get_count_tweet_seen() + agent_i.get_count_tweet_written() == 1:
                agent_i.outdate_persona_memory()
            if agent_j.get_count_tweet_seen() + agent_j.get_count_tweet_written() == 1:
                agent_j.outdate_persona_memory()

            row.append(agent_j.current_belief)
            agent_j_previos_interaction_type = agent_j.previous_interaction_type

            if agent_j_previos_interaction_type in ["none", "write", "read"]:
                tweet_j = agent_j.produce_tweet(
                    previous_interaction_type=agent_j_previos_interaction_type,
                    tweet_written_count=agent_j.get_count_tweet_written(),
                    add_to_memory=False,
                )
                agent_j.increase_count_tweet_written()
                row.append(tweet_j)
            else:
                raise ValueError(
                    "agent_j_previos_interaction_type is not valid: {}".format(
                        agent_j_previos_interaction_type
                    )
                )

            agent_j.add_to_memory(
                tweet_written=tweet_j,
                previos_interaction_type=agent_j_previos_interaction_type,
                current_interaction_type="write",
                tweet_written_count=agent_j.get_count_tweet_written(),
            )

            row.append(agent_i.agent_name)
            agent_i_pre_belief = agent_i.current_belief
            row.append(agent_i_pre_belief)

            agent_i_previos_interaction_type = agent_i.previous_interaction_type
            if agent_i_previos_interaction_type in ["none", "write", "read"]:
                response_i = agent_i.receive_tweet(
                    tweet_j,
                    previous_interaction_type=agent_i_previos_interaction_type,
                    tweet_written_count=agent_i.get_count_tweet_written(),
                    add_to_memory=False,
                )
                agent_i.increase_count_tweet_seen()
                row.append(response_i)
            else:
                raise ValueError(
                    "agent_i_previos_interaction_type is not valid: {}".format(
                        agent_i_previos_interaction_type
                    )
                )

            agent_i.add_to_memory(
                tweet_seen=tweet_j,
                response=response_i,
                previos_interaction_type=agent_i_previos_interaction_type,
                current_interaction_type="read",
                tweet_written_count=agent_i.get_count_tweet_written(),
            )

            agent_i.current_belief = extract_belief(response_i)
            agent_i_post_belief = agent_i.current_belief
            row.append(agent_i_post_belief)
            row.append(agent_i_post_belief - agent_i_pre_belief)
            writer.writerow(row)

            agent_i.previous_interaction_type = "read"
            agent_j.previous_interaction_type = "write"

            for agent in list_agents:
                dict_csv[agent.agent_name].append(agent.current_belief)

            dict_agent_tweet[agent_j].append((t + 1, agent_j.current_belief, tweet_j))
            dict_agent_response[agent_i].append((t + 1, response_i))

        pd.DataFrame.from_dict(dict_csv).to_csv(out_name)
    return (
        post_process_memory(list_agents, path_result, date_version),
        post_process_tweet(dict_agent_tweet, path_result, date_version),
        post_process_response(dict_agent_response, path_result, date_version),
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
            + "_reflection.txt",
        )
        with open(out_name, "w+") as f:
            f.write(agent.memory.prompt.messages[0].prompt.template)
            f.write("\n------------------------------\n")
            for i in range(len(agent.memory.memory.chat_memory.messages)):
                f.write(agent.memory.memory.chat_memory.messages[i].content)
                f.write("\n------------------------------\n")

    return


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
        + "_agent_tweet_history_"
        + args.distribution
        + "_reflection.csv"
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
        + "_agent_response_history_"
        + args.distribution
        + "_reflection.csv"
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
    if not rating_flag:
        tweet = tweet.replace("5-", "")
        belief = int(re.findall("[+-]?\d", tweet)[-1])
    else:
        belief = 0
    return belief


def extract_reasoning(tweet):
    if not rating_flag:
        reasoning = tweet.split("\nFinal Answer")[0]
    else:
        reasoning = "Reasoning:" + tweet.split("\nReasoning")[-1]
    return reasoning


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
    date_version = "20231212"

    global rating_flag
    rating_flag = True if "no_rating" in args.version_set else False
    rating_flag = args.no_rating

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
