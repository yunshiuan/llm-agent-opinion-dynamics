"""
Create a random sample of agents' responses from the 15 topics.
- this will be the sample for human annotation (using a 5-point scale) to compare with the FLAN-T5-XXL's responses
- stratified sampling: 15 topics, 2 framings, 3 confirmation bias levels, 2 memory levels
"""
import pandas as pd
import re
from os.path import join
import os
import json
from collections import Counter

class ResponseSampler:
  def __init__(self, list_file_response_csv, list_llm_rating_csv):
    """
    This class takes in a list of file paths to the csv files of the agents' responses and a list of file paths to the csv files of the FLAN-T5-XXL's ratings. It reads and preprocesses the responses and the ratings. It then stratified samples the responses and saves the sampled responses to a csv file.
    - this will be the sample for human annotation (using a 5-point scale) to compare with the (FLAN-T5-XXL's) ratings
    - stratified sampling: 15 topics, 2 framings, 3 confirmation bias levels, 2 memory levels        

    Args:
      list_file_response_csv (list of str): list of file paths to the csv files of the agents' responses
      list_llm_rating_csv (list of str): list of file paths to the csv files of the FLAN-T5-XXL's ratings
    """
    # set a random seed
    # np.random.seed(0)

    assert len(list_file_response_csv) == len(list_llm_rating_csv)
    # ----------------
    # read and preprocess the responses
    # ----------------
    list_df_responses = []
    for file_response_csv in list_file_response_csv:
      df_responses = pd.read_csv(file_response_csv)
      # get the topic, framing, confirmation bias, memory level from the file name
      # - prompt version
      prompt_v = re.search(r"_(v\d+.*)_agent_response",
                           file_response_csv).group(1)
      # - topic
      topic_v = re.search(r"_(v\d+)_", file_response_csv).group(1)
      topic_name = DICT_TOPIC_NAME[topic_v]
      df_responses["prompt_version"] = prompt_v
      df_responses["topic_version"] = topic_v
      df_responses["topic_name"] = topic_name
      # - framing
      if "reverse" in file_response_csv:
        framing = "TRUE"
      else:
        framing = "FALSE"
      df_responses["framing"] = framing
      # - confirmation bias
      if "default" in file_response_csv:
        confirmation_bias = "default"
      elif "strong_confirmation_bias" in file_response_csv:
        confirmation_bias = "strong"
      elif ("confirmation_bias" in file_response_csv) and ("strong" not in file_response_csv):
        confirmation_bias = "weak"
      else:
        raise ValueError(
            f"Unknown confirmation bias level in {file_response_csv}")
      df_responses["confirmation_bias"] = confirmation_bias
      # - memory level
      if "reflection" in file_response_csv:
        memory_level = "reflective"
      else:
        memory_level = "cumulative"
      df_responses["memory_level"] = memory_level
      list_df_responses.append(df_responses)
    self.df_responses = pd.concat(list_df_responses, ignore_index=True)
    # ----------------
    # select and derive columns of interest
    # ----------------
    self.df_responses = self.df_responses[[
        "prompt_version", "topic_version", "topic_name", "framing", "confirmation_bias", "memory_level",
        "Agent Name", "Original Belief",
        # to get the series of response
        "Response chain",
        # save the corresponding time step
        "Belief Changes Time Step"]].copy()
    # ----------------
    # parse the response chain into a list of strings
    # ----------------
    list_parsed_response = []
    for i_row in range(self.df_responses.shape[0]):
      response_chain = self.df_responses.loc[i_row, "Response chain"]
      # Remove leading and trailing brackets and quotes
      cleaned_string = response_chain[2:-2]
      # Regular expression pattern to split the string
      # It looks for ', ", |', |",
      pattern = r"', \", |', |\", "
      # Split the string using the regular expression pattern
      parsed_list = re.split(pattern, cleaned_string)
      list_parsed_response.append(parsed_list)
    # save the parsed list to the dataframe as a new column
    self.df_responses.loc[:, "Response chain parsed"] = pd.Series(
        list_parsed_response)

    # ----------------
    # parse the belief change time step into a list of integers
    # ----------------
    # use json load to parse the string into a list
    list_parsed_belief_change_time_step = []
    for i_row in range(self.df_responses.shape[0]):
      belief_change_time_step = self.df_responses.loc[i_row,
                                                      "Belief Changes Time Step"]
      parsed_list = json.loads(belief_change_time_step)
      list_parsed_belief_change_time_step.append(parsed_list)
    # save the parsed list to the dataframe as a new column
    self.df_responses.loc[:, "Belief Changes Time Step parsed"] = pd.Series(
        list_parsed_belief_change_time_step)

    # ----------------
    # sanity check
    # ----------------
    # the parsed response chain and belief change time step should have the same length
    for i_row in range(self.df_responses.shape[0]):
      parsed_response_chain = self.df_responses.loc[i_row,
                                                    "Response chain parsed"]
      parsed_belief_change_time_step = self.df_responses.loc[i_row,
                                                             "Belief Changes Time Step parsed"]
      assert len(parsed_response_chain) == len(parsed_belief_change_time_step), \
          f"parsed response chain and belief change time step have different length at row {i_row}"

    # ----------------
    # read and preprocess the LLM's ratings
    # ----------------
    list_df_llm_rating = []
    for file_llm_rating_csv in list_llm_rating_csv:
      df_llm_rating = pd.read_csv(file_llm_rating_csv)
      # get the topic, framing, confirmation bias, memory level from the file name
      # - topic
      topic_v = re.search(
          r"_(v\d+)_", file_llm_rating_csv).group(1)
      topic_name = DICT_TOPIC_NAME[topic_v]
      df_llm_rating["topic_version"] = topic_v
      df_llm_rating["topic_name"] = topic_name
      # - framing
      if "reverse" in file_llm_rating_csv:
        framing = "TRUE"
      else:
        framing = "FALSE"
      df_llm_rating["framing"] = framing
      # - confirmation bias
      if "default" in file_llm_rating_csv:
        confirmation_bias = "default"
      elif "strong_confirmation_bias" in file_llm_rating_csv:
        confirmation_bias = "strong"
      elif ("confirmation_bias" in file_llm_rating_csv) and ("strong" not in file_llm_rating_csv):
        confirmation_bias = "weak"
      else:
        raise ValueError(
            f"Unknown confirmation bias level in {file_llm_rating_csv}")
      df_llm_rating["confirmation_bias"] = confirmation_bias
      # - memory level
      if "reflection" in file_llm_rating_csv:
        memory_level = "reflective"
      else:
        memory_level = "cumulative"
      df_llm_rating["memory_level"] = memory_level
      list_df_llm_rating.append(df_llm_rating)
    self.df_llm_rating = pd.concat(
        list_df_llm_rating, ignore_index=True)
    # ----------------
    # reorder the columns
    # - move all added columns to the start
    # ----------------
    list_col = list(self.df_llm_rating)
    list_col_to_move = ["topic_version", "topic_name",
                        "framing", "confirmation_bias", "memory_level"]
    for col_to_move in list_col_to_move:
      list_col.remove(col_to_move)
    list_col = list_col_to_move + list_col
    self.df_llm_rating = self.df_llm_rating[list_col]
    # ----------------
    # sanity check
    # ----------------
    # the number of rows
    # - +1: add the t=0 step
    assert self.df_llm_rating.shape[0] == (N_TIME_STEPS+1) * len(list_file_response_csv), \
        f"the number of rows in the ground truth rating dataframe is not correct. Expected: {(N_TIME_STEPS+1) * len(list_file_response_csv)}, Actual: {self.df_llm_rating.shape[0]}"

  def stratified_sample(self, file_output_sample, list_stratum=["topic_name", "framing", "confirmation_bias", "memory_level"], n_agent_chain_per_stratum=1, n_response_per_agent_chain=1, random_seed=666):
    """
    Stratified sampling from the responses. The stratum is defined by the list of columns. The number of agent's response chains to sample per stratum is defined by n_agent_chain_per_stratum. The number of responses to sample per agent's response chain is defined by n_response_per_agent_chain. The total number of responses sampled is n_agent_chain_per_stratum * n_response_per_agent_chain * number of stratum.

    Args:
      list_stratum (list of str): list of column names to stratify the responses. Default: ["topic_name", "framing", "confirmation_bias", "memory_level"].
      n_agent_chain_per_stratum (int): number of agent' response chains to sample per stratum. Default: 1.
      n_response_per_agent_chain (int): number of responses to sample per agent's response chain. Default: 1.
      random_seed (int): random seed. Default: 666.
    """
    # ----------------
    # stratified sampling: sample n_agent_chain_per_stratum * n_response_per_agent_chain responses from each stratum
    # ----------------
    # group by the stratum
    df_grouped = self.df_responses.groupby(list_stratum)
    # sample chains from each stratum
    list_df_sampled = []
    list_response_sampled = []
    list_time_step_sampled = []
    for name, group in df_grouped:
      df_sampled = group.sample(n_agent_chain_per_stratum, random_state=random_seed)
      for index, row in df_sampled.iterrows():
        parsed_response_chain = row["Response chain parsed"]
        parsed_belief_change_time_step = row["Belief Changes Time Step parsed"]
        for i_response in range(n_response_per_agent_chain):
          response_sampled = parsed_response_chain[i_response]
          belief_change_time_step_sampled = parsed_belief_change_time_step[i_response]
          list_df_sampled.append(df_sampled)
          list_response_sampled.append(response_sampled)
          list_time_step_sampled.append(
              belief_change_time_step_sampled)

    df_sampled = pd.concat(list_df_sampled, ignore_index=True)
    df_sampled["Response sampled"] = pd.Series(list_response_sampled)
    df_sampled["Belief Changes Time Step sampled"] = pd.Series(
        list_time_step_sampled)

    # ----------------
    # add the corresponding true rating (e.g., FLAN-T5-XXL's) of the sampled responses
    # ----------------
    list_llm_rating = []
    for row_index, row in df_sampled.iterrows():
      # get the topic, framing, confirmation bias, memory level
      topic_this = row["topic_version"]
      framing_this = row["framing"]
      confirmation_bias_this = row["confirmation_bias"]
      memory_level_this = row["memory_level"]
      # get the time step and the agent name
      time_step_this = row["Belief Changes Time Step sampled"]
      agent_name_this = row["Agent Name"]
      # get the corresponding ground truth rating from the ground truth rating dataframe
      # - get the relevant rows
      df_truth_rating_this = self.df_llm_rating.loc[
          (self.df_llm_rating["topic_version"] == topic_this) &
          (self.df_llm_rating["framing"] == framing_this) &
          (self.df_llm_rating["confirmation_bias"] == confirmation_bias_this) &
          (self.df_llm_rating["memory_level"] == memory_level_this)]
      # - get the rating
      rating_this = df_truth_rating_this[agent_name_this].to_list()[
          time_step_this]
      list_llm_rating.append(rating_this)
    df_sampled["LLM Rating"] = pd.Series(list_llm_rating)

    # ----------------
    # add a column of the wording of the topic statement
    # ----------------
    # - using DICT_TOPIC_FRAMING_STATEMENT
    list_topic_framing_statement = []
    for row_index, row in df_sampled.iterrows():
      topic_this = row["topic_version"]
      framing_this = row["framing"]
      topic_framing_statement_this = DICT_TOPIC_FRAMING_STATEMENT[(
          topic_this, framing_this)]
      list_topic_framing_statement.append(topic_framing_statement_this)
    df_sampled["Statement"] = pd.Series(
        list_topic_framing_statement)
    # ----------------
    # reorder the columns
    # - move all the chain columns to the end (raw and parsed)
    # ----------------
    list_col = list(df_sampled.columns)
    list_col_to_move = ["Response chain", "Response chain parsed",
                        "Belief Changes Time Step", "Belief Changes Time Step parsed"]
    for col_to_move in list_col_to_move:
      list_col.remove(col_to_move)
    list_col.extend(list_col_to_move)
    df_sampled = df_sampled[list_col]

    # ----------------
    # sanity check
    # ----------------
    # the size of the sampled dataframe should be n_agent_chain_per_stratum * n_response_per_agent_chain * number of stratum
    assert df_sampled.shape[0] == n_agent_chain_per_stratum * n_response_per_agent_chain * df_grouped.ngroups, \
        f"the size of the sampled dataframe is not correct. Expected: {n_agent_chain_per_stratum * n_response_per_agent_chain * df_grouped.ngroups}, Actual: {df_sampled.shape[0]}"
    # each stratum should have n_agent_chain_per_stratum * n_response_per_agent_chain
    assert df_sampled.groupby(list_stratum).size().unique() == n_agent_chain_per_stratum * n_response_per_agent_chain, \
        f"each stratum should have {n_agent_chain_per_stratum * n_response_per_agent_chain} responses"
    # rename the output file
    file_output_sample = file_output_sample.replace(
        "N_SAMPLE", str(df_sampled.shape[0]))

    # ----------------
    # also save the distribution of LLM's rating
    # ----------------
    # get the distribution of true rating
    df_grouped = df_sampled.groupby(["LLM Rating"])
    # get both the count and percentage
    df_grouped_count = df_grouped.size().reset_index(name="count")
    df_grouped_count["percentage"] = df_grouped_count["count"] / \
        df_grouped_count["count"].sum()
    # sort by count
    df_grouped_count.sort_values(
        by=["count"], ascending=False, inplace=True)
    # save the print out to a txt file
    file_output_count = file_output_sample.replace(
        ".csv", "_llm_rating_dist.csv")
    df_grouped_count.to_csv(file_output_count, index=False)

    # ----------------
    # save the sampled dataframe
    # ----------------
    # shuffle the rows
    df_sampled = df_sampled.sample(frac=1, random_state=random_seed)
    # add response_id (unique id for each response) as the first column
    df_sampled.insert(0, "response_id", range(1, 1 + len(df_sampled)))    
    df_sampled.to_csv(file_output_sample, index=False)


class FileResponseRatingRetriever:
  def __init__(self, path_input_response, path_input_llm_rating, list_topic_version, list_framing, list_confirmation_bias, list_memory_level=["reflective", "cumulative"]):
    """
    Retrieve the list of files of the agents' responses and the list of files of the LLM's ratings.

    Args:
      path_input_response (str): path to the folder containing the csv files of the agents' responses
      path_input_llm_rating (str): path to the folder containing the csv files of the LLM's ratings
      list_topic_version (list of str): list of topic versions
      list_framing (list of str): list of framings
      list_confirmation_bias (list of str): list of confirmation bias levels
      list_memory_level (list of str): list of memory levels
    Returns:
      df_files (pd.DataFrame): dataframe with the following columns:
        - file_response_csv (str): file name of the agents' response
        - file_llm_rating_csv (str): file name of the LLM's rating
    """
    self.path_input_response = path_input_response
    self.path_input_llm_rating = path_input_llm_rating
    self.list_topic_version = list_topic_version
    self.list_framing = list_framing
    self.list_confirmation_bias = list_confirmation_bias
    self.list_memory_level = list_memory_level

    # ----------------
    # get the list of file for the agents' responses and the LLM's ratings
    # ----------------
    list_file_response_csv = []
    list_file_llm_rating_csv = []
    for topic_version in self.list_topic_version:
      for framing in self.list_framing:
        for confirmation_bias in self.list_confirmation_bias:
          for memory_level in self.list_memory_level:
            # ----------------
            # specify naming variables
            # ----------------
            if memory_level == "reflective":
              response_var_memory = "_reflection"
            elif memory_level == "cumulative":
              response_var_memory = ""
            else:
              raise ValueError(f"Unknown memory level {memory_level}")
            if framing == "TRUE":
              response_var_framing = "_reverse"
            elif framing == "FALSE":
              response_var_framing = ""
            else:
              raise ValueError(f"Unknown framing {framing}")

            # ----------------
            # get the files for the agents' responses
            # ----------------
            regex_pattern = f"seed1_10_100_{topic_version}_{confirmation_bias}{response_var_framing}_(\d+)_agent_response_history_uniform{response_var_memory}.csv"
            # e.g.,  "seed1_10_100_v37_default_20231109_agent_response_history_uniform_reflection.csv", "seed1_10_100_v37_strong_confirmation_bias_reverse_20231108_agent_response_history_uniform.csv"

            # find all files matching the regex pattern in `self.path_input_response`
            file_response_csv_this = [f for f in os.listdir(self.path_input_response) if re.match(
                regex_pattern, f)]
            # if there is more than one file, chech if they only differ by the date
            if len(file_response_csv_this) > 1:
              # see if, after removing the date, the files are the same
              file_response_csv_this_no_date = [
                  re.sub(r"_(\d{8})_agent_response", "", f) for f in file_response_csv_this]
              if len(set(file_response_csv_this_no_date)) == 1:
                # if so, keep the file name with the newest date
                file_response_csv_this = [
                    sorted(file_response_csv_this)[-1]]
              else:
                raise ValueError(
                    f"More than one file matching the regex pattern {regex_pattern} in {self.path_input_response}")
            assert len(file_response_csv_this) == 1
            file_response_csv_this = file_response_csv_this[0]
            # return the full path
            file_response_csv_this = join(
                self.path_input_response, file_response_csv_this)
            list_file_response_csv.append(file_response_csv_this)

            # ----------------
            # get the files for the LLM's ratings
            # ----------------
            # ----------------
            # specify naming variables
            # ----------------
            if memory_level == "reflective":
              rating_var_memory = "reflection_"
            elif memory_level == "cumulative":
              rating_var_memory = ""
            else:
              raise ValueError(f"Unknown memory level {memory_level}")
            if framing == "TRUE":
              rating_var_framing = "_reverse"
            elif framing == "FALSE":
              rating_var_framing = ""
            else:
              raise ValueError(f"Unknown framing {framing}")

            # ----------------
            # get the files for the LLM's ratings
            # ----------------
            regex_pattern = f"{rating_var_memory}seed1_10_100_{topic_version}_{confirmation_bias}{response_var_framing}_(\d+)_flan-t5-xxl_uniform.csv"
            # e.g., "reflection_seed1_10_100_v37_default_20231201_flan-t5-xxl_uniform.csv", "seed1_10_100_v37_strong_confirmation_bias_reverse_20231201_flan-t5-xxl_uniform.csv"
            # find all files matching the regex pattern in `self.path_input_llm_rating`
            file_llm_rating_csv_this = [f for f in os.listdir(self.path_input_llm_rating) if re.match(
                regex_pattern, f)]
            if len(file_llm_rating_csv_this) > 1:
              # see if, after removing the date, the files are the same
              file_llm_rating_csv_this_no_date = [
                  re.sub(r"_(\d{8})_flan-t5-xxl_uniform", "", f) for f in file_llm_rating_csv_this]
              if len(set(file_llm_rating_csv_this_no_date)) == 1:
                # if so, keep the file name with the newest date
                file_llm_rating_csv_this = [
                    sorted(file_llm_rating_csv_this)[-1]]
              else:
                raise ValueError(
                    f"More than one file matching the regex pattern {regex_pattern} in {self.path_input_llm_rating}")
            assert len(file_llm_rating_csv_this) == 1
            file_llm_rating_csv_this = file_llm_rating_csv_this[0]
            # return the full path
            file_llm_rating_csv_this = join(
                self.path_input_llm_rating, file_llm_rating_csv_this)
            list_file_llm_rating_csv.append(file_llm_rating_csv_this)

    self.list_file_llm_rating_csv = list_file_llm_rating_csv
    self.list_file_response_csv = list_file_response_csv

    # ----------------
    # sanity check
    # ----------------
    # the number of unique files
    expected_num_files = len(self.list_topic_version) * len(self.list_framing) * \
        len(self.list_confirmation_bias) * len(self.list_memory_level)
    assert len(set(self.list_file_response_csv)) == expected_num_files, \
        f"the number of files is not correct. Expected: {expected_num_files}, Actual: {len(self.list_file_response_csv)}"
    assert len(set(self.list_file_llm_rating_csv)) == expected_num_files, \
        f"the number of files is not correct. Expected: {expected_num_files}, Actual: {len(self.list_file_llm_rating_csv)}"
    df_files = pd.DataFrame(
        {"file_response_csv": self.list_file_response_csv, "file_llm_rating_csv": self.list_file_llm_rating_csv})
    self.df_files = df_files

  def get_df_files(self):
    """
    Return the dataframe with the following columns:
      - file_response_csv (str): file name of the agents' response
      - file_llm_rating_csv (str): file name of the LLM's rating
    """
    return self.df_files


if __name__ == "__main__":
  # - par
  DEBUG = False
  MODEL = "gpt-3.5-turbo-16k"
  # MODEL = "gpt-4-1106-preview"
  # MODEL = "vicuna-33b-v1.3"
  N_AGENT_CHAIN_PER_STRATUM = 1
  N_RESPONSE_PER_AGENT_CHAIN = 1
  N_SAMPLE = N_AGENT_CHAIN_PER_STRATUM * N_RESPONSE_PER_AGENT_CHAIN
  N_TIME_STEPS = 100
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
  # (topic_version, framing)
  DICT_TOPIC_FRAMING_STATEMENT = \
      {("v37", "TRUE"): "theory XYZ that claims that the earth is an irregularly shaped ellipsoid rather than flat",
       ("v37", "FALSE"): "the theory XYZ that claims that the Earth is flat",
       ("v38", "TRUE"): "theory XYZ that claims that the Tyrannosaurus Rex and humans did not co-exist on Earth at the same",
       ("v38", "FALSE"): "theory XYZ that claims that the Tyrannosaurus Rex and humans co-existed on Earth at the same time",
       ("v39", "TRUE"): "theory XYZ that claims that it is not possible for humans to communicate with the dead",
       ("v39", "FALSE"): "theory XYZ that claims that it is possible for humans to communicate with the dead",
       ("v40", "TRUE"): "theory XYZ that claims that it is not possible to predict someone’s future by looking at their palm characteristics",
       ("v40", "FALSE"): "theory XYZ that claims that it is possible to predict someone’s future by looking at their palm characteristics",
       ("v41", "TRUE"): "theory XYZ that claims that global warming is a real phenomenon and global climate is rapidly growing warmer",
       ("v41", "FALSE"): "theory XYZ that claims that global warming is a conspiracy by governments worldwide and is not a real phenomenon",
       ("v52", "TRUE"): "Theory XYZ that claims that US astronauts have landed on the moon",
       ("v52", "FALSE"): "Theory XYZ that claims that US astronauts have not landed on the moon",
       ("v53", "TRUE"): "Theory XYZ that claims that the twin towers were not brought down from the inside by explosives during the 9/11 attack",
       ("v53", "FALSE"): "Theory XYZ that claims that the twin towers were brought down from the inside by explosives during the 9/11 attack",
       ("v54", "TRUE"): "Theory XYZ that the Nazi government in Germany murdered approximately 6 million Jewish people during the second world war",
       ("v54", "FALSE"): "Theory XYZ that the Nazi government in Germany did not murder approximately 6 million Jewish people during the second world war",
       ("v55", "TRUE"): "Theory XYZ that claims that the US unemployment rate in 2016 was lower than 40%",
       ("v55", "FALSE"): "Theory XYZ that claims that the US unemployment rate in 2016 was higher than 40%",
       ("v56", "TRUE"): "Theory XYZ that claims that Barack Obama was born in Hawaii",
       ("v56", "FALSE"): "Theory XYZ that claims that Barack Obama was born in Kenya",
       ("v57", "TRUE"): "Theory XYZ that claims that a bicycle usually has two wheels",
       ("v57", "FALSE"): "Theory XYZ that claims that a bicycle usually has four wheels",
       ("v58", "TRUE"): "Theory XYZ that claims that Washington DC is in the United States",
       ("v58", "FALSE"): "Theory XYZ that claims that Washington DC is not in the United States",
       ("v59", "TRUE"): "Theory XYZ that claims that human beings are not born with a brain",
       ("v59", "FALSE"): "Theory XYZ that claims that human beings are born with a brain",
       ("v60", "TRUE"): "Theory XYZ that claims that fire is hot",
       ("v60", "FALSE"): "Theory XYZ that claims that fire is cold",
       ("v61", "TRUE"): "Theory XYZ that claims that on a clear sunny day, the sky is usually blue",
       ("v61", "FALSE"): "Theory XYZ that claims that on a clear sunny day, the sky is usually red"
       }
  topics = [key[0] for key in DICT_TOPIC_FRAMING_STATEMENT.keys()]
  framings = [key[1] for key in DICT_TOPIC_FRAMING_STATEMENT.keys()]
  topic_counts = Counter(topics)
  framing_counts = Counter(framings)
  assert len(set(topic_counts.values())) == 1, "Not all topics have the same count"
  assert len(set(framing_counts.values())) == 1, "Not all framings have the same count"  

  LIST_FRAMING = ["TRUE", "FALSE"]
  LIST_CONFIRMATION_BIAS = ["default",
                            "confirmation_bias", "strong_confirmation_bias"]
  LIST_MEMORY = ["reflective", "cumulative"]

  DATE_OUTPUT = "20240112"
  # - path
  PATH_INPUT_RESPONSE = join("results/opinion_dynamics/Flache_2017", MODEL)
  PATH_INPUT_LLM_RATING = join("final_csv_files")
  PATH_OUTPUT = join("results/opinion_dynamics/Flache_2017",
                     MODEL, "sample_for_human_annotation")
  # - file
  # FILE_INPUT = join(PATH_INPUT, "all_prompt_metrics.txt")
  FILE_OUTPUT = join(
      PATH_OUTPUT, "df_sampled_n_N_SAMPLE_{}.csv".format(DATE_OUTPUT))

  # ----------------
  # get the list of the input files
  # ----------------
  if DEBUG:
    LIST_FILE_RESPONSE_CSV = [
        join(PATH_INPUT_RESPONSE,
             "seed1_10_100_v37_default_20231109_agent_response_history_uniform_reflection.csv"),
        join(PATH_INPUT_RESPONSE,
             "seed1_10_100_v37_strong_confirmation_bias_reverse_20231108_agent_response_history_uniform.csv")
    ]
    LIST_FILE_LLM_RATING_CSV = [
        join(PATH_INPUT_LLM_RATING,
             "reflection_seed1_10_100_v37_default_20231201_flan-t5-xxl_uniform.csv"),
        join(PATH_INPUT_LLM_RATING,
             "seed1_10_100_v37_strong_confirmation_bias_reverse_20231201_flan-t5-xxl_uniform.csv")
    ]
  else:
    file_resoinse_rating_retriever = FileResponseRatingRetriever(
        path_input_response=PATH_INPUT_RESPONSE,
        path_input_llm_rating=PATH_INPUT_LLM_RATING,
        list_topic_version=list(DICT_TOPIC_NAME.keys()),
        list_framing=LIST_FRAMING,
        list_confirmation_bias=LIST_CONFIRMATION_BIAS,
        list_memory_level=LIST_MEMORY)
    df_files = file_resoinse_rating_retriever.get_df_files()
    LIST_FILE_RESPONSE_CSV = df_files["file_response_csv"].to_list()
    LIST_FILE_LLM_RATING_CSV = df_files["file_llm_rating_csv"].to_list()
  # ----------------
  # sample the responses
  # ----------------
  response_sampler = ResponseSampler(
      list_file_response_csv=LIST_FILE_RESPONSE_CSV,
      list_llm_rating_csv=LIST_FILE_LLM_RATING_CSV)
  response_sampler.stratified_sample(
      list_stratum=["topic_name", "framing",
                    "confirmation_bias", "memory_level"],
      n_agent_chain_per_stratum=N_AGENT_CHAIN_PER_STRATUM,
      n_response_per_agent_chain=N_RESPONSE_PER_AGENT_CHAIN,
      file_output_sample=FILE_OUTPUT, 
      random_seed=666)
