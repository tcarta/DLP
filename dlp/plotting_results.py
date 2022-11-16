import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_log(dir_):
    """Loads log from a directory and adds it to a list of dataframes."""
    df = pd.read_csv(os.path.join(dir_, 'log.csv'),
                     on_bad_lines='warn')
    if not len(df):
        print("empty df at {}".format(dir_))
        return
    df['model'] = dir_
    return df


def load_logs(root):
    dfs = []
    for root, dirs, files in os.walk(root, followlinks=True):
        for file_ in files:
            if file_ == 'log.csv':
                dfs.append(load_log(root))
    return dfs


def plot_average_impl(df, regexps, labels, limits, colors, y_value='return_mean', window=10, agg='mean',
                      x_value='frames'):
    """Plot averages over groups of runs  defined by regular expressions."""
    df = df.dropna(subset=[y_value])
    unique_models = df['model'].unique()
    model_groups = [[m for m in unique_models if re.match(regex, m)]
                    for regex in regexps]

    for regex, models, label, color in zip(regexps, model_groups, labels, colors):
        # print("regex: {}".format(regex))
        df_re = df[df['model'].isin(models)]
        # the average doesn't make sense if most models are not included,
        # so we only for the period of training that has been done by all models
        num_frames_per_model = [df_model[x_value].max()
                                for _, df_model in df_re.groupby('model')]
        for _, df_model in df_re.groupby('model'):
            print(df_model[x_value].max())
        median_progress = sorted(num_frames_per_model)[(len(num_frames_per_model) - 1) // 2]
        mean_duration = np.mean([
            df_model['duration'].max() for _, df_model in df_re.groupby('model')])
        df_re = df_re[df_re[x_value] <= limits]

        # smooth
        parts = []
        for _, df_model in df_re.groupby('model'):
            df_model = df_model.copy()
            df_model.loc[:, y_value] = df_model[y_value].rolling(window).mean()
            parts.append(df_model)
        df_re = pd.concat(parts)
        df_agg = df_re.groupby([x_value]).mean()
        # df_max = df_re.groupby([x_value]).max()[y_value]
        # df_min = df_re.groupby([x_value]).min()[y_value]
        values = df_agg[y_value]
        std = df_re.groupby([x_value]).std()[y_value]
        # print(std.iloc[-1])
        df_max = values + std
        df_min = values - std

        # pyplot.plot(df_agg.index, values, label='{} SE: {}'.format(label, round(values.sum()/len(values), 3)))
        print(("{} last mean:{} last std: {}").format(label, values.iloc[-1], std.iloc[-1]))
        plt.plot(df_agg.index, values, label=label, color=color)
        # pyplot.plot(df_agg.index, values, label=label)
        plt.fill_between(df_agg.index, df_max, df_min, alpha=0.25, color=color)
        print(regex, median_progress, mean_duration / 86400.0, values.iloc[-1])
        print("{} sample efficiency: {}".format(label, values.sum() / len(values)))


dfs = load_logs('/home/tcarta/DLP/storage')
df = pd.concat(dfs, sort=True)


def plot_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='return_mean', *args, **kwargs)
    # plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11}, bbox_to_anchor=(1.1, 1.1))
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Reward", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    # plt.figure(figsize=(8, 6), dpi=100)
    plt.show()


def plot_sucess_rate_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='success_rate', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Success Rate", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_entropy_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='entropy', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Entropy", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_policy_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='policy_loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Policy Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_value_loss_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='value_loss', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Value Loss", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_grad_norm_average(*args, **kwargs):
    """Plot averages over groups of runs  defined by regular expressions."""
    plt.figure(figsize=(7.5, 5))
    plot_average_impl(y_value='grad_norm', *args, **kwargs)
    plt.legend(handlelength=0.5, handleheight=0.5, prop={"size": 11})
    plt.xlabel("Frames", fontsize=15)

    plt.title("Average Grad Norm", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

# ####################### Performance function of the size of the LLM ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large', 'FLAN-T5-small', 'Classic-A2C']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green']
# plot_average(df, regexs, labels, limits, colors)
# plot_sucess_rate_average(df, regexs, labels, limits, colors)

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-large', 'FLAN-T5-small']
limits = 400000
colors = ['tab:blue', 'tab:orange']
# plot_entropy_average(df, regexs, labels, limits, colors)
# plot_loss_average(df, regexs, labels, limits, colors)
# plot_policy_loss_average(df, regexs, labels, limits, colors)
# plot_value_loss_average(df, regexs, labels, limits, colors)
# plot_grad_norm_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the number of actions ######################## #

# ####################### LLM_small ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_9_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-small 3 actions', 'FLAN-T5-small 9 actions']
limits = 400000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### LLM_large ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_9_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-large 3 actions', 'FLAN-T5-large 9 actions']
limits = 400000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### LLM_mixt ######################## #

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_9_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_9_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-small 3 actions', 'FLAN-T5-small 9 actions',
          'FLAN-T5-large 3 actions', 'FLAN-T5-large 9 actions']
limits = 400000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey']
# plot_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the number of rooms ######################## #
# ####################### LLM_large ######################## #
# ####################### 1 room ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large 1 room', 'Classic A2C 1 room']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### 2 rooms ######################## #
regexs = ['.*llm_gtm_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTM-PPO-NoPre.*']
labels = ['FLAN-T5-large 2 rooms', 'Classic A2C 2 rooms']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### 4 rooms ######################## #
regexs = ['.*llm_gtlarge_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTLarge-PPO-NoPre.*']
labels = ['FLAN-T5-large 4 rooms', 'Classic A2C 4 rooms']
limits = 200000
colors = ['tab:blue', 'tab:orange']
plot_average(df, regexs, labels, limits, colors)

# ####################### Mixt ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtm_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtlarge_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-large 1 room', 'FLAN-T5-large 2 rooms', 'FLAN-T5-large 4 rooms']
limits = 200000
colors = ['tab:blue', 'tab:orange', 'tab:grey']
plot_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the number of distractors ######################## #
# ####################### LLM_large ######################## #
# ####################### 4 distractors ######################## #
regexs = ['.*llm_gtl_distractor_4_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL4-PPO-NoPre.*']
labels = ['FLAN-T5-large 4 distractors', 'Classic A2C 4 distractors']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### 8 distractors ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large 8 distractors', 'Classic A2C 8 distractors']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### 16 distractors ######################## #
regexs = ['.*llm_gtl_distractor_16_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL16-PPO-NoPre.*']
labels = ['FLAN-T5-large 16 distractors', 'Classic A2C 16 distractors']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_average(df, regexs, labels, limits, colors)

# ####################### Mixt ######################## #
regexs = ['.*llm_gtl_distractor_4_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_distractor_16_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-large 4 distractors', 'FLAN-T5-large 8 distractors', 'FLAN-T5-large 16 distractors']
limits = 200000
colors = ['tab:blue', 'tab:orange', 'tab:grey']
# plot_average(df, regexs, labels, limits, colors)

# ####################### Full ######################## #
regexs = ['.*llm_gtl_distractor_4_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_distractor_16_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL4-PPO-NoPre.*',
          '.*GTL-PPO-NoPre.*',
          '.*GTL16-PPO-NoPre.*']
labels = ['FLAN-T5-large 4 distractors', 'FLAN-T5-large 8 distractors', 'FLAN-T5-large 16 distractors',
          'Classic A2C 4 distractors', 'Classic A2C 8 distractors', 'Classic A2C 16 distractors']
limits = 200000
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey', 'tab:cyan', 'tab:purple']
# plot_average(df, regexs, labels, limits, colors)

# ####################### Performance function of the type of reward ######################## #
# ####################### LLM_large ######################## #
regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_simple_env_reward_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large', 'FLAN-T5-large-simple-reward', 'Classic-A2C']
limits = 200000
colors = ['tab:blue', 'tab:orange', 'tab:green']
# plot_average(df, regexs, labels, limits, colors)

regexs = ['.*llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_simple_env_reward_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*']
labels = ['FLAN-T5-large', 'FLAN-T5-large-simple-reward', 'Classic-A2C']
limits = 200000
colors = ['tab:blue', 'tab:orange']
# plot_loss_average(df, regexs, labels, limits, colors)
# plot_policy_loss_average(df, regexs, labels, limits, colors)
# plot_value_loss_average(df, regexs, labels, limits, colors)
# plot_entropy_average(df, regexs, labels, limits, colors)

# ####################### Ablation: pretraining of the LLM large ######################## #
# ####################### LLM_large ######################## #
regexs = ['.*llm_gtl_simple_env_reward_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*llm_gtl_nbr_env_32_Flan_T5large_untrained_nbr_actions_3_shape_reward_beta_0_seed.*',
          '.*GTL-PPO-NoPre.*']
labels = ['FLAN-T5-large-simple-reward', 'FLAN-T5-large-untrained', 'Classic-A2C']
limits = 250000
colors = ['tab:blue', 'tab:orange', 'tab:green']
# plot_average(df, regexs, labels, limits, colors)
# plot_loss_average(df, regexs, labels, limits, colors)



# ####################### Distribution shift study 3 actions ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_3_shape_reward_beta_0_seed_2',
             'llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_3_shape_reward_beta_0_seed_2']

legend = ['T5_large_1',
          'T5_large_2',
          'T5_small_1',
          'T5_small_2'
          ]

columns_names=['{}'.format(i) for i in range(3*6)]
indices = np.arange(3)
actions = ["turn left", "turn right", "go forward"]
width = 0.1
for i in range(len(name_file)):
    for j in range(4):
        distrib_large = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[i]+"/distrib.csv", names=columns_names)
        # p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][:3], width=width, alpha=0.5, label="update: {}".format(j*50))
        p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][12:15], width=width, alpha=0.5, label="update: {}".format(j*50))
        plt.xticks(indices, actions)
        plt.legend()
    plt.title(legend[i])
    plt.show()"""

# ####################### Distribution shift study 3 actions untrained ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5large_untrained_nbr_actions_3_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5large_untrained_nbr_actions_3_shape_reward_beta_0_seed_2']

legend = ['T5_large_untrained_1',
          'T5_untrained_large_2']

columns_names=['{}'.format(i) for i in range(3*6)]
indices = np.arange(3)
actions = ["turn left", "turn right", "go forward"]
width = 0.1
for i in range(len(name_file)):
    for j in range(4):
        distrib_large = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[i]+"/distrib.csv", names=columns_names)
        # p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][:3], width=width, alpha=0.5, label="update: {}".format(j*50))
        p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][12:15], width=width, alpha=0.5, label="update: {}".format(j*50))
        plt.xticks(indices, actions)
        plt.legend()
    plt.title(legend[i])
    plt.show()"""

# ####################### Distribution shift study 9 actions ######################## #

"""name_file = ['llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_9_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5small_nbr_actions_9_shape_reward_beta_0_seed_2',
             'llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_9_shape_reward_beta_0_seed_1',
             'llm_gtl_nbr_env_32_Flan_T5large_nbr_actions_9_shape_reward_beta_0_seed_2']

legend = ['T5_small_1',
          'T5_small_2',
          'T5_large_1',
          'T5_large_2']

columns_names=['{}'.format(i) for i in range(9*6)]
indices = np.arange(9)
actions = ["turn left", "turn right", "go forward", "eat", "dance", "sleep", "do nothing", "cut", "think"]
width = 0.1
for i in range(len(name_file)):
    for j in range(2):
        distrib_large = pd.read_csv("/home/tcarta/DLP/storage/logs/"+name_file[i]+"/distrib.csv", names=columns_names)
        # p = plt.bar(indices-0.05+0.1*j, distrib_large.iloc[j][:9], width=width, alpha=0.5, label="update: {}".format(j*50))
        p = plt.bar(indices-0.15+0.1*j, distrib_large.iloc[j][36:45], width=width, alpha=0.5, label="update: {}".format(j*50))
        plt.xticks(indices, actions, rotation=25)
        plt.legend()
    plt.title(legend[i])
    plt.show()"""