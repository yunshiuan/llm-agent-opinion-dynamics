# # ------------------------------
# # chatgpt, v37-v41, v57-v61
# # ------------------------------
# MODEL=gpt-3.5-turbo-16k
# DATE_PICKLE=20240124
# STR_LIST_PROMPT="37 38 39 40 41 52 53 54 55 56 57 58 59 60 61"

# # python scripts/every_topic_metrics.py -pv 37 38 39 40 41 52 53 54 55 56 57 58 59 60 61
# python scripts/every_topic_metrics.py -pv ${STR_LIST_PROMPT} -input_date ${DATE_PICKLE} --model ${MODEL} --output_file "all_prompt_metrics_cumulative_${MODEL}_${DATE_PICKLE}.txt"
# python scripts/every_topic_metrics.py -pv ${STR_LIST_PROMPT} -input_date ${DATE_PICKLE} --model ${MODEL} --output_file "all_prompt_metrics_reflective_${MODEL}_${DATE_PICKLE}.txt" --reflection


# # ------------------------------
# # gpt4, v37-v41
# # ------------------------------
# MODEL=gpt-4-1106-preview
# DATE_PICKLE=20240124
# STR_LIST_PROMPT="37 38 39 40 41"

# # python scripts/every_topic_metrics.py -pv 37 38 39 40 41 52 53 54 55 56 57 58 59 60 61
# python scripts/every_topic_metrics.py -pv ${STR_LIST_PROMPT} -input_date ${DATE_PICKLE} --model ${MODEL} --output_file "all_prompt_metrics_cumulative_${MODEL}_${DATE_PICKLE}.txt"
# python scripts/every_topic_metrics.py -pv ${STR_LIST_PROMPT} -input_date ${DATE_PICKLE} --model ${MODEL} --output_file "all_prompt_metrics_reflective_${MODEL}_${DATE_PICKLE}.txt" --reflection

# ------------------------------
# vicuna-33b, v37-v41, v57-v61, reflective only
# ------------------------------
MODEL=vicuna-33b-v1.3
DATE_PICKLE=20240124
STR_LIST_PROMPT="37 38 39 40 41 52 53 54 55 56 57 58 59 60 61"

# python scripts/every_topic_metrics.py -pv 37 38 39 40 41 52 53 54 55 56 57 58 59 60 61
python scripts/every_topic_metrics.py -pv ${STR_LIST_PROMPT} -input_date ${DATE_PICKLE} --model ${MODEL} --output_file "all_prompt_metrics_reflective_${MODEL}_${DATE_PICKLE}.txt" --reflection