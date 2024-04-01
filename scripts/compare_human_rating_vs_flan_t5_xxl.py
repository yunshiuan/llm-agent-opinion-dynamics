"""
Compute the agreement matrix between human ratings and FLAN-T5-XXL's ratings.
- This script processes two sets of ratings: one from humans and the other from FLAN-T5-XXL model.
- It calculates the agreement matrix to understand how closely these two sets of ratings align.
- Additionally, the script computes intercoder reliability, using a statistical measure like Cohen's Kappa, to evaluate the consistency of ratings between humans and FLAN-T5-XXL.
- The script is intended for use in opinion dynamics research, where understanding the alignment and reliability of different rating sources is crucial.

The script works with the following structure:
- Input: 
  - FLAN-T5-XXL's ratings (CSV format)
  - Human ratings (CSV format)
- Output:
  - Agreement matrix
  - Intercoder reliability measures
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from os.path import join
import krippendorff


class RatingComparer:
  def __init__(self, file_rating_llm, file_rating_human):
    """
    Initialize the RatingComparer with paths to the FLAN-T5-XXL ratings and human ratings.

    Args:
        file_rating_llm (str): Path to the FLAN-T5-XXL ratings.
        file_rating_human (str): Path to the human ratings.
    """
    self.df_rating_llm = pd.read_csv(file_rating_llm)
    self.df_rating_human = pd.read_csv(file_rating_human)

    # --------------
    # merge llm rating to human rating as a new column (using response_id as the key)
    # --------------
    self.df_rating_human = pd.merge(self.df_rating_human, self.df_rating_llm[[
                                    "response_id", "LLM Rating"]], on='response_id', how='left')
    # check if LLM Rating (NEW)  LLM Rating are the same
    assert all(self.df_rating_human["LLM Rating"]
               == self.df_rating_human["LLM Rating (NEW)"])

    # --------------
    # compute the majority vote for each reponse for human ratings
    # --------------
    LIST_RATERS = ["sean", "nikunj", "sid"]
    LIST_COL_NAME_RATERS = [f"rating_{rater}" for rater in LIST_RATERS]
    self.raters = LIST_RATERS
    self.list_col_name_raters = LIST_COL_NAME_RATERS
    # compute the majority vote for each reponse, if there is no majority, then use sean's rating
    for i_row, row in self.df_rating_human.iterrows():
      # get the ratings from all raters
      list_ratings = [row[col_name] for col_name in LIST_COL_NAME_RATERS]
      # count the number of votes for each rating
      # - if there are multiple votes with the same count, then use sean's rating
      dict_count = {}
      for rating in list_ratings:
        if rating not in dict_count:
          dict_count[rating] = 1
        else:
          dict_count[rating] += 1
      # get the majority vote
      majority_vote = None
      max_count = 0
      for rating in dict_count:
        if dict_count[rating] > max_count:
          majority_vote = rating
          max_count = dict_count[rating]
      # if there is no majority, then use sean's rating
      if max_count == 1:
        majority_vote = row["rating_sean"]

      # update the majority vote
      self.df_rating_human.at[i_row, "rating_majority"] = majority_vote

  def compute_agreement_matrix(self, file_ouptput_agreement_matrix):
    """
    Compute the agreement matrix between FLAN-T5-XXL's ratings and human ratings.

    Args:
        file_ouptput_agreement_matrix (str): Path to the agreement matrix.
    """
    # --------------
    # compute the agreement matrix between LLM rating and the majority vote
    # --------------
    # initialize the agreement matrix
    agreement_matrix = pd.crosstab(
        self.df_rating_human['LLM Rating'], self.df_rating_human['rating_majority'])
    agreement_matrix = agreement_matrix.reset_index()
    # rename the columns
    col_name_new = ['LLM Rating'] + ["rating_majority_" +
                                     str(rating) for rating in agreement_matrix.columns.tolist()[1:]]
    agreement_matrix.columns = col_name_new
    # --------------
    # add a total column and a total row
    # --------------        
    # add a total column
    agreement_matrix["total"] = agreement_matrix.sum(axis=1)
    # add a total row
    agreement_matrix.loc['total'] = agreement_matrix.sum(axis=0)
    # Save the agreement matrix to a CSV file
    agreement_matrix.to_csv(file_ouptput_agreement_matrix)

  def calculate_intercoder_reliability(self, file_ouptput_intercoder_reliability):
    """
    Calculate intercoder reliability (e.g., Cohen's Kappa).

    Args:
        file_ouptput_intercoder_reliability (str): Path to the intercoder reliability file.
    """
    # Transpose the DataFrame subset to get raters in columns and responses in rows
    data = self.df_rating_human[self.list_col_name_raters].transpose().values

    # Calculate Krippendorff's alpha
    alpha = krippendorff.alpha(data)

    # create a df saving the intercoder reliability
    df_intercoder_reliability = pd.DataFrame(
        {"Intercoder Reliability": [alpha]})
    # Save the intercoder reliability 
    df_intercoder_reliability.to_csv(file_ouptput_intercoder_reliability)



if __name__ == "__main__":

  # - par
  MODEL = "gpt-3.5-turbo-16k"
  # - path
  PATH_SAMPLE_FOR_HUMAN_ANNOTATION = f'results/opinion_dynamics/Flache_2017/{MODEL}/sample_for_human_annotation/'
  # - file
  FILE_FLAN_T5_XXL_RATINGS = join(
      PATH_SAMPLE_FOR_HUMAN_ANNOTATION, 'df_sampled_n_180_20240112.csv')
  FILE_HUMAN_RATINGS = join(PATH_SAMPLE_FOR_HUMAN_ANNOTATION,
                            'df_sampled_n_180_top_100_20240112_human_ratings.csv')
  FILE_OUPTPUT_AGREEMENT_MATRIX = join(
      PATH_SAMPLE_FOR_HUMAN_ANNOTATION, 'agreement_matrix_n_180_top_100_20240112.csv')
  FILE_OUPTPUT_INTERCODER_RELIABILITY = join(
      PATH_SAMPLE_FOR_HUMAN_ANNOTATION, 'intercoder_agreement_n_180_top_100_20240112.csv')

  comparer = RatingComparer(FILE_FLAN_T5_XXL_RATINGS, FILE_HUMAN_RATINGS)
  comparer.compute_agreement_matrix(FILE_OUPTPUT_AGREEMENT_MATRIX)
  comparer.calculate_intercoder_reliability(
      FILE_OUPTPUT_INTERCODER_RELIABILITY)
