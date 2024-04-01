# 1. Summarize the per-topic metrics into result tables.
# 2. Statistical tests for the result tables.

library(dplyr)
library(tidyr)
library(stringr)
library(car)
library(lme4)

# -----------
# Constants
# -----------
# - Parameters
# MODEL_NAME <- "gpt-3.5-turbo-16k"
# MODEL_NAME <- "gpt-4-1106-preview"
MODEL_NAME <- "vicuna-33b-v1.3"
NUM_AGENTS <- 10
NUM_TIME_STEPS <- 100
if (MODEL_NAME == "gpt-3.5-turbo-16k") {
  NUM_TOPICS_SCIENCE <- 5
  NUM_TOPICS_HISTORY <- 5
  NUM_TOPICS_COMMON <- 5
} else if (MODEL_NAME == "gpt-4-1106-preview") {
  NUM_TOPICS_SCIENCE <- 5
  NUM_TOPICS_HISTORY <- 0
  NUM_TOPICS_COMMON <- 0
} else if (MODEL_NAME == "vicuna-33b-v1.3") {
  NUM_TOPICS_SCIENCE <- 5
  NUM_TOPICS_HISTORY <- 5
  NUM_TOPICS_COMMON <- 5
}
NUM_TOPICS_ALL <- NUM_TOPICS_SCIENCE + NUM_TOPICS_HISTORY + NUM_TOPICS_COMMON
NUM_FRAMINGS <- 2
LIST_TOPIC_TYPES <- c("science", "history", "common")
if (MODEL_NAME %in% c("gpt-3.5-turbo-16k", "gpt-4-1106-preview")) {
  LIST_MEMORY <- c("Cumulative", "Reflective")
  if (MODEL_NAME == "gpt-3.5-turbo-16k") {
    INPUT_FILE_NAME <- "all_prompt_metrics_merged_gpt-3.5-turbo-16k_20240124.csv"
    DATE_OUTPUT_VERSION <- "_20240124"
  } else if (MODEL_NAME == "gpt-4-1106-preview") {
    INPUT_FILE_NAME <- "all_prompt_metrics_merged_gpt-4-1106-preview_20240124.csv"
    DATE_OUTPUT_VERSION <- "_20240124"
  }
  LIST_CONFIRMATION_BIAS <- c("default", "confirmation_bias", "strong_confirmation_bias")
  LIST_CONFIRMATION_BIAS_IN_ANOVA <- c("default", "confirmation_bias", "strong_confirmation_bias")
} else if (MODEL_NAME == "vicuna-33b-v1.3") {
  LIST_MEMORY <- c("Reflective")
  INPUT_FILE_NAME <- "all_prompt_metrics_reflective_vicuna-33b-v1.3_20240124.csv"
  DATE_OUTPUT_VERSION <- ""
  LIST_CONFIRMATION_BIAS <- c("default",
                              # "control",
                              "confirmation_bias", "strong_confirmation_bias")
  LIST_CONFIRMATION_BIAS_IN_ANOVA <- c("default", "confirmation_bias", "strong_confirmation_bias")
}
NUM_MEMORIES <- length(LIST_MEMORY)
NUM_LEVELS_CONFIRMATION_BIAS <- length(LIST_CONFIRMATION_BIAS)
LIST_METRICS <- c("bias_mean", "bias_se", "diversity_mean", "diversity_se", "n")
LIST_METRICS_X_MEMORY <-
  expand_grid(LIST_METRICS, LIST_MEMORY) %>%
  mutate(col_name = paste0(LIST_METRICS, "_", LIST_MEMORY)) %>%
  pull(col_name)

# - Paths
PATH_ROOT <- "/Users/vimchiz/github_local/PhD_projects/llm-opinion-dynamics"
PATH_INPUT_DATA <- file.path(PATH_ROOT, "results", "opinion_dynamics", "Flache_2017", MODEL_NAME, "summary_table")
PATH_OUTPUT_TABLE <- file.path(PATH_ROOT, "results", "opinion_dynamics", "Flache_2017", MODEL_NAME, "summary_table")
# - Files
FILE_INPUT_DATA <- file.path(PATH_INPUT_DATA, INPUT_FILE_NAME)
FILE_OUTPUT_TABLE_ALL_TOPICS <- file.path(
  PATH_OUTPUT_TABLE,
  paste0(
    "table_topic_all",
    DATE_OUTPUT_VERSION, ".csv"
  )
)
FILE_OUTPUT_TEST_BIAS_FALSE_FRAMING <- file.path(
  PATH_OUTPUT_TABLE,
  paste0(
    "test_bias_under_false_framing",
    DATE_OUTPUT_VERSION, ".txt"
  )
)
FILE_OUTPUT_TEST_BIAS_TRUE_FRAMING <- file.path(
  PATH_OUTPUT_TABLE,
  paste0(
    "test_bias_under_true_framing",
    DATE_OUTPUT_VERSION, ".txt"
  )
)
FILE_OUTPUT_TEST_BIAS_DIFF_FRAMING <- file.path(
  PATH_OUTPUT_TABLE,
  paste0(
    "test_bias_diff_framing",
    DATE_OUTPUT_VERSION, ".txt"
  )
)
FILE_OUTPUT_TEST_DIVERSITY_ANOVA_CONF_BIAS <- file.path(
  PATH_OUTPUT_TABLE,
  paste0(
    "test_diversity_anova_conf_bias",
    DATE_OUTPUT_VERSION, ".txt"
  )
)

# -----------
# Read and Process Data
# -----------
df_all_prompt_metrics <- read.csv(FILE_INPUT_DATA, stringsAsFactors = FALSE)
# make memory, framing, confirmation bias level factors
df_all_prompt_metrics <- df_all_prompt_metrics %>%
  mutate(
    memory = factor(memory, levels = LIST_MEMORY),
    framing = factor(framing, levels = c("FALSE", "TRUE")),
    confirmation_bias = factor(confirmation_bias,
      levels = LIST_CONFIRMATION_BIAS
    )
  )
# - sanity check
stopifnot(nrow(df_all_prompt_metrics) == NUM_TOPICS_ALL * NUM_MEMORIES * NUM_FRAMINGS * NUM_LEVELS_CONFIRMATION_BIAS)
# -- unique topics
stopifnot(length(unique(df_all_prompt_metrics$topic_name)) == NUM_TOPICS_ALL)
# -- unique memories
stopifnot(length(unique(df_all_prompt_metrics$memory)) == NUM_MEMORIES)
# -- unique framings
stopifnot(length(unique(df_all_prompt_metrics$framing)) == NUM_FRAMINGS)
# -- unique confirmation bias levels
stopifnot(length(unique(df_all_prompt_metrics$confirmation_bias)) == NUM_LEVELS_CONFIRMATION_BIAS)

# -----------
# Summary Table 1: Global summary across all topics
# -----------
# - Group by memory, framing, confirmation bias level
# - compute mean and standard error of "bias" and "diversity"
df_summary_global <- df_all_prompt_metrics %>%
  group_by(memory, framing, confirmation_bias) %>%
  summarize(
    bias_mean = mean(bias),
    bias_se = sd(bias) / sqrt(n()),
    diversity_mean = mean(diversity),
    diversity_se = sd(diversity) / sqrt(n()),
    n = n()
  )
# - reshape from long to wide (the memory column is made wider)

df_summary_global <- df_summary_global %>%
  # pivot_wider(names_from = memory, values_from = c(bias_mean, bias_se, diversity_mean, diversity_se, n)) %>%
  pivot_wider(names_from = memory, values_from = all_of(LIST_METRICS)) %>%
  as.data.frame() %>%
  dplyr::select(framing, confirmation_bias, all_of(LIST_METRICS_X_MEMORY)) %>%
  # dplyr::select(framing, confirmation_bias, bias_mean_Cumulative, bias_se_Cumulative, diversity_mean_Cumulative, diversity_se_Cumulative, bias_mean_Reflective, bias_se_Reflective, diversity_mean_Reflective, diversity_se_Reflective, n_Cumulative, n_Reflective) %>%
  # round to 3 decimal places for numbers
  mutate_if(is.numeric, round, 2)
# - sanity check
stopifnot(nrow(df_summary_global) == NUM_FRAMINGS * NUM_LEVELS_CONFIRMATION_BIAS)
# -- check if n_Cumulative ==  n_Reflective == NUM_TOPICS_ALL
stopifnot(all(df_summary_global$n_Cumulative == NUM_TOPICS_ALL))
stopifnot(all(df_summary_global$n_Reflective == NUM_TOPICS_ALL))
# - write to file
write.csv(df_summary_global, file = FILE_OUTPUT_TABLE_ALL_TOPICS, row.names = FALSE)

# -----------
# Summary Table 2: Summary for each topic category
# -----------
# - Group by memory, framing, confirmation bias level, topic category
for (topic_type_this in LIST_TOPIC_TYPES) {
  # - filter by topic type
  df_all_prompt_metrics_topic_type <- df_all_prompt_metrics %>%
    filter(topic_type == topic_type_this)
  # - compute mean and standard error of "bias" and "diversity"
  df_summary_topic_type <- df_all_prompt_metrics_topic_type %>%
    group_by(memory, framing, confirmation_bias) %>%
    summarize(
      bias_mean = mean(bias),
      bias_se = sd(bias) / sqrt(n()),
      diversity_mean = mean(diversity),
      diversity_se = sd(diversity) / sqrt(n()),
      n = n()
    )
  # - reshape from long to wide (the memory column is made wider)
  df_summary_topic_type <- df_summary_topic_type %>%
    pivot_wider(names_from = memory, values_from = all_of(LIST_METRICS)) %>%
    as.data.frame() %>%
    dplyr::select(framing, confirmation_bias, all_of(LIST_METRICS_X_MEMORY)) %>%
    # round to 3 decimal places for numbers
    mutate_if(is.numeric, round, 2)
  # - sanity check
  stopifnot(nrow(df_summary_topic_type) == NUM_FRAMINGS * NUM_LEVELS_CONFIRMATION_BIAS)
  # -- check if n_Cumulative ==  n_Reflective == NUM_TOPICS_TOPIC_TYPE
  if (topic_type_this == "science") {
    NUM_TOPICS_TOPIC_TYPE <- NUM_TOPICS_SCIENCE
  } else if (topic_type_this == "history") {
    NUM_TOPICS_TOPIC_TYPE <- NUM_TOPICS_HISTORY
  } else if (topic_type_this == "common") {
    NUM_TOPICS_TOPIC_TYPE <- NUM_TOPICS_COMMON
  }
  stopifnot(all(df_summary_topic_type$n_Cumulative == NUM_TOPICS_TOPIC_TYPE))
  stopifnot(all(df_summary_topic_type$n_Reflective == NUM_TOPICS_TOPIC_TYPE))
  # - write to file
  FILE_OUTPUT_TABLE_TOPIC_TYPE <- file.path(
    PATH_OUTPUT_TABLE,
    paste0("table_topic_", topic_type_this, DATE_OUTPUT_VERSION, ".csv")
  )
  write.csv(df_summary_topic_type, file = FILE_OUTPUT_TABLE_TOPIC_TYPE, row.names = FALSE)
}

# -----------
# Statistical Test 1: Whether bias is negative under false framing (for each confirmation bias levels)
# - one t test for cumulative memory and another t test for reflective memory
# -----------
sink(FILE_OUTPUT_TEST_BIAS_FALSE_FRAMING)

for (confirmation_bias_this in LIST_CONFIRMATION_BIAS){
  print("--------------------------------------------------")
  print(paste0("Test whether bias is negative under false framing for confirmation bias level: ", confirmation_bias_this))
  print("--------------------------------------------------")  
  for (memory_this in LIST_MEMORY) {
    # - filter by memory
    list_bias_this <-
      df_all_prompt_metrics %>%
      filter(memory == memory_this) %>%
      filter(
        framing == "FALSE",
        confirmation_bias == confirmation_bias_this
      ) %>%
      pull(bias)
    # - sanity check
    stopifnot(length(list_bias_this) == NUM_TOPICS_ALL)
    # - t test
    t_test_result <- t.test(list_bias_this)
    # - print result
    print(paste0("memory: ", memory_this))
    print(t_test_result)
  }
}
sink()
closeAllConnections()


# -----------
# Statistical Test 2: Whether positive is positive under true framing (for each confirmation bias levels)
# - one t test for cumulative memory and another t test for reflective memory
# -----------
sink(FILE_OUTPUT_TEST_BIAS_TRUE_FRAMING)
print("Test whether bias is positive under true framing")
for (confirmation_bias_this in LIST_CONFIRMATION_BIAS){
  print("--------------------------------------------------")
  print(paste0("Test whether bias is positive under true framing for confirmation bias level: ", confirmation_bias_this))
  print("--------------------------------------------------")  
  for (memory_this in LIST_MEMORY) {
    # - filter by memory
    list_bias_this <-
      df_all_prompt_metrics %>%
      filter(memory == memory_this) %>%
      filter(
        framing == "TRUE",
        confirmation_bias == confirmation_bias_this
      ) %>%
      pull(bias)
    # - sanity check
    stopifnot(length(list_bias_this) == NUM_TOPICS_ALL)
    # - t test
    t_test_result <- t.test(list_bias_this)
    # - print result
    print(paste0("memory: ", memory_this))
    print(t_test_result)
  }
}
sink()
closeAllConnections()


# -----------
# Statistical Test 3: Whether the bias is "more negative" under false framing than true framing (for each confirmation bias)
# - paired t test
# -----------
sink(FILE_OUTPUT_TEST_BIAS_DIFF_FRAMING)
for (confirmation_bias_this in LIST_CONFIRMATION_BIAS){
  print("--------------------------------------------------")  
  print(paste0("Test whether bias is different under false framing versus true framing for confirmation bias level: ", confirmation_bias_this))
  print("--------------------------------------------------")
  for (memory_this in LIST_MEMORY) {
    # - filter by memory
    list_bias_false_framing <-
      df_all_prompt_metrics %>%
      filter(memory == memory_this) %>%
      filter(
        framing == "FALSE",
        confirmation_bias == confirmation_bias_this
      ) %>%
      pull(bias)
    list_bias_true_framing <-
      df_all_prompt_metrics %>%
      filter(memory == memory_this) %>%
      filter(
        framing == "TRUE",
        confirmation_bias == confirmation_bias_this
      ) %>%
      pull(bias)
    # - sanity check
    stopifnot(length(list_bias_false_framing) == NUM_TOPICS_ALL)
    stopifnot(length(list_bias_true_framing) == NUM_TOPICS_ALL)
    # - piared t test
    t_test_result <- t.test(list_bias_false_framing, list_bias_true_framing, paired = TRUE)
    # - print result
    print(paste0("memory: ", memory_this))
    print(t_test_result)
  }
}
sink()
closeAllConnections()

# -----------
# Statistical Test 4: Whether the a stronger confirmation bias correlates with a larger diversity
# - 1-way ANOVA comparing the three levels of confirmation bias; one ANOVA per memory type
# - diversity~confirmation_bias
# -----------
NUM_CONFIRMATION_BIAS_IN_ANOVA <- length(LIST_CONFIRMATION_BIAS_IN_ANOVA)
sink(FILE_OUTPUT_TEST_DIVERSITY_ANOVA_CONF_BIAS)

print("Test whether a stronger confirmation bias correlates with a larger diversity")
for (memory_this in LIST_MEMORY) {
  # - filter by memory
  df_all_prompt_metrics_memory_this <-
    df_all_prompt_metrics %>%
    filter(memory == memory_this) %>%
    # ignore the control condition in this analysis
    filter(confirmation_bias != "control") %>%
    mutate(confirmation_bias = factor(confirmation_bias, levels = LIST_CONFIRMATION_BIAS_IN_ANOVA))
  # - sanity check
  stopifnot(nrow(df_all_prompt_metrics_memory_this) == NUM_TOPICS_ALL * NUM_FRAMINGS * NUM_CONFIRMATION_BIAS_IN_ANOVA)
  # - Setting up planned contrasts
  # Assuming levels of confirmation_bias are in the order: 'default', 'confirmation_bias', 'strong_confirmation_bias'
  contrasts(df_all_prompt_metrics_memory_this$confirmation_bias) <-
    matrix(
      c(
        -2, 1, 1,
        0, -1, 1
      ),
      ncol = 2
    )
  # - 1-way ANOVA with contrasts
  anova_result <- lm(diversity ~ confirmation_bias, data = df_all_prompt_metrics_memory_this, contrasts = list(confirmation_bias = contrasts(df_all_prompt_metrics_memory_this$confirmation_bias)))
  # - type 3 ANOVA
  f_test_anova_result <- Anova(anova_result, type = "III")

  # - planned contrasts analysis
  # - contrast 1: default vs confirmation_bias
  planed_constrast <- summary.lm(anova_result)

  # - print result
  print(paste0("memory: ", memory_this))
  print("ANOVA result")
  print(f_test_anova_result)
  print("Planned contrast result")
  print(contrasts(df_all_prompt_metrics_memory_this$confirmation_bias))
  print(planed_constrast)
}
sink()
closeAllConnections()
