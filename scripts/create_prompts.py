import os
from os.path import join
import argparse

parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Script")
parser.add_argument(
    "-p",
    "--prompt_version",
    default=52,
    type=int,
    help="Prompt Version",
)
args = parser.parse_args()

list_directories = [f"v{args.prompt_version}_default", f"v{args.prompt_version}_default_reverse", 
                    f"v{args.prompt_version}_confirmation_bias", f"v{args.prompt_version}_confirmation_bias_reverse", 
                    f"v{args.prompt_version}_strong_confirmation_bias", f"v{args.prompt_version}_strong_confirmation_bias_reverse"]

file_names = ["step1_persona.md", "step1_persona_past.md", 
             "step2_produce_tweet_prev_none.md", "step2_produce_tweet_prev_read.md", "step2_produce_tweet_prev_tweet.md",
             "step2b_add_to_memory_prev_none_cur_read.md", "step2b_add_to_memory_prev_none_cur_tweet.md", 
             "step2b_add_to_memory_prev_read_cur_read.md", "step2b_add_to_memory_prev_read_cur_tweet.md",
             "step2b_add_to_memory_prev_tweet_cur_read.md", "step2b_add_to_memory_prev_tweet_cur_tweet.md",
             "step3_receive_tweet_prev_none.md", "step3_receive_tweet_prev_read.md", "step3_receive_tweet_prev_tweet.md"]

