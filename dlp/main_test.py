"""
This script run a simple agent in a BabyAI GoTo-Local environment.
"""
import os
import sys
import csv
import json
import logging

import time
import numpy as np
import torch
import gym
import babyai.utils as utils
import hydra
import test_llm

from babyai.paral_env_simple import ParallelEnv
from colorama import Fore
from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction
from accelerate import Accelerator

lamorel_init()
logger = logging.getLogger(__name__)
accelerator = Accelerator()


# TODO add the value of the true reward *20 who should receive the final reward?
def reward_function(subgoal_proba=None, reward=None, policy_value=None, llm_0=None):
    if reward > 0:
        return [20 * reward, 0]
    else:
        return [0, 0]


# TODO think about a correct value for the beta of the reward shaping part
def reward_function_shapped(subgoal_proba=None, reward=None, policy_value=None, llm_0=None):
    if reward > 0:
        return [20 * reward - np.log(subgoal_proba / policy_value), -np.log(subgoal_proba / policy_value)]
    else:
        return [-1 - np.log(subgoal_proba / policy_value), -1 - np.log(subgoal_proba / policy_value)]


class ValueModuleFn(BaseModuleFunction):

    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

    def initialize(self):
        llm_hidden_size = self.llm_config.to_dict()[self.llm_config.attribute_map['hidden_size']]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][0][0, len(tokenized_context["input_ids"]) - 1, :]
        else:
            model_head = forward_outputs['encoder_last_hidden_state'][0, len(tokenized_context["input_ids"]) - 1, :]

        value = self.value_head_op(model_head.to(self.device))
        return value.cpu()


"""dict_modifier_english = [{},
                         {
                             'key': 'chair',
                             'ball': 'table',
                             'box': 'car'
                         },
                         {
                             'red': 'vermilion',
                             'green': 'jade',
                             'blue': 'cyan',
                             'purple': 'violet',
                             'yellow': 'golden',
                             'grey': 'silver'
                         },
                         {
                             'key': 'dax',
                             'ball': 'xolo',
                             'box': 'afze'
                         },
                         {
                             'red': 'faze',
                             'green': 'jatu',
                             'blue': 'croh',
                             'purple': 'vurst',
                             'yellow': 'gakul',
                             'grey': 'sil'
                         },
                         {
                             'key': 'dax',
                             'ball': 'xolo',
                             'box': 'afze',
                             'red': 'faze',
                             'green': 'jatu',
                             'blue': 'croh',
                             'purple': 'vurst',
                             'yellow': 'gakul',
                             'grey': 'sil'
                         },
                         {
                             "to": "face",
                             "Observation": "viewing",
                             "the": "some",
                             "A": "One",
                             ":": "",
                             "go": ""
                         }]

dict_modifier_french = [{},
                        {
                            'clef': 'chaise',
                            'balle': 'table',
                            'boîte': 'voiture'
                        },
                        {
                            'rouge': 'vermilion',
                            'verte': 'jade',
                            'bleue': 'cyan',
                            'violette': 'mauve',
                            'jaune': 'dorée',
                            'gris': 'argent'
                        },
                        {
                            'clef': 'dax',
                            'balle': 'xolo',
                            'boîte': 'afze'
                        },
                        {
                            'rouge': 'faze',
                            'verte': 'jatu',
                            'bleue': 'croh',
                            'violette': 'vurst',
                            'jaune': 'gakul',
                            'grise': 'sil'
                        },
                        {
                            'clef': 'dax',
                            'balle': 'xolo',
                            'boîte': 'afze',
                            'rouge': 'faze',
                            'verte': 'jatu',
                            'bleue': 'croh',
                            'violette': 'vurst',
                            'jaune': 'gakul',
                            'grise': 'sil'
                        }]
dict_dict_modifier = {'english': dict_modifier_english, 'french': dict_modifier_french}
dict_modifier_name = ['no_modifications', 'other_name_same_categories', 'adj_synonym', 'no_meaning_nouns',
                      'no_meaning_adj', 'no_meaning_words', 'important_words_suppress']"""

dict_modifier_english = [{"go to": "reach"},
                         {"go to": "face"},
                         {"Goal of the agent": "I would like the agent to"},
                         {"Goal of the agent": "You have to"}]
dict_modifier_name = ['predicate_synonym_reach', 'predicate_synonym_face', "change_intro_first_personne_speaker",
                      "change_intro_first_personne_agent"]
dict_dict_modifier = {'english': dict_modifier_english}


class updater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, "is_loaded"):
            self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                        "/" + kwargs["id_expe"] + "/model.checkpoint"))
            self.is_loaded = True


def run_agent(args, algo):
    format_str = ("Language: {} | Name dict: {} | Episodes Done: {} | Reward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) |\
     Success Rate: {: .2f} | \nReshaped: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Bonus: {: .2f} +- {: .2f}\
                                 (Min: {: .2f} Max: {: .2f})")

    dm = dict_dict_modifier[args.language]
    for d, d_name in zip(dm, dict_modifier_name):
        logs = algo.generate_trajectories(d, args.language)

        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        reshaped_return_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        reshaped_return_bonus_per_episode = utils.synthesize(logs["reshaped_return_bonus_per_episode"])
        # num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [args.language, d_name, logs['episodes_done'], *return_per_episode.values(),
                success_per_episode['mean'],
                *reshaped_return_per_episode.values(),
                *reshaped_return_bonus_per_episode.values()]

        logger.info(Fore.YELLOW + format_str.format(*data) + Fore.RESET)


# This will be overriden by lamorel's launcher if used
@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # lm server
    lm_server = Caller(config_args.lamorel_args, custom_updater_class=updater,
                       custom_module_functions={'value': ValueModuleFn(config_args.lamorel_args.llm_args.model_type)})

    id_expe = config_args.rl_script_args.name_experiment + \
              '_nbr_env_{}_'.format(config_args.rl_script_args.number_envs) + \
              '{}_'.format(config_args.rl_script_args.name_model) + \
              'nbr_actions_{}_'.format(config_args.rl_script_args.size_action_space) + \
              'shape_reward_beta_{}_'.format(config_args.rl_script_args.reward_shaping_beta) + \
              'seed_{}'.format(config_args.rl_script_args.seed)

    if not config_args.rl_script_args.zero_shot:
        lm_server.update([None, None, None, None], [[None], [None], [None], [None]],
                         id_expe=id_expe, saving_path_model=config_args.rl_script_args.saving_path_model)

    # Env
    name_env = config_args.rl_script_args.name_environment
    seed = config_args.rl_script_args.seed
    envs = []
    subgoals = []
    number_envs = config_args.rl_script_args.number_envs
    for i in range(number_envs):
        env = gym.make(name_env)
        env.seed(int(1e9 * seed + i))
        envs.append(env)
        if config_args.rl_script_args.size_action_space == 3:
            subgoals.append(["turn left", "turn right", "go forward"])

        elif config_args.rl_script_args.size_action_space == 6:
            subgoals.append(["turn left", "turn right", "go forward",
                             "do nothing", "cut", "think"])

        else:
            subgoals.append(["turn left", "turn right", "go forward",
                             "eat", "dance", "sleep",
                             "do nothing", "cut", "think"])

    envs = ParallelEnv(envs)

    if config_args.rl_script_args.reward_shaping_beta == 0:
        reshape_reward = reward_function
    else:
        reshape_reward = reward_function_shapped  # TODO ad the beta

    id_expe = config_args.rl_script_args.name_experiment + \
              '_nbr_env_{}_'.format(config_args.rl_script_args.number_envs) + \
              '{}_'.format(config_args.rl_script_args.name_model) + \
              'nbr_actions_{}_'.format(config_args.rl_script_args.size_action_space) + \
              'shape_reward_beta_{}_'.format(config_args.rl_script_args.reward_shaping_beta) + \
              'seed_{}'.format(config_args.rl_script_args.seed)

    model_path = os.path.join(config_args.rl_script_args.saving_path_model, id_expe)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    algo = test_llm.BaseAlgo(envs, lm_server, config_args.rl_script_args.number_episodes, reshape_reward,
                             subgoals)
    run_agent(config_args.rl_script_args, algo)
    lm_server.close()


if __name__ == '__main__':
    main()
