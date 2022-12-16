"""
This script run a simple agent in a BabyAI GoTo-Local environment.
"""
import os
import csv
import json
from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)
from colorama import Fore

import time

import numpy as np
import torch
import gym
import torch.nn.functional as F
from torch.distributions import Categorical

import babyai.rl
import babyai.utils as utils
from babyai.paral_env_simple import ParallelEnv

from agents.drrn.drrn import DRRN_Agent

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction

lamorel_init()

import hydra

from accelerate import Accelerator

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
        return [0 - np.log(subgoal_proba / policy_value), 0 - np.log(subgoal_proba / policy_value)]


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
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][0][0, len(tokenized_context["input_ids"]) - 1, :]
        else:
            model_head = forward_outputs['encoder_last_hidden_state'][0, len(tokenized_context["input_ids"]) - 1, :]

        value = self.value_head_op(model_head.to(self.device))
        return value.cpu()

class ActionHeadsModuleFn(BaseModuleFunction):
    def __init__(self, model_type, action_space_size):
        super().__init__()
        self._model_type = model_type
        self._action_space_size = action_space_size

    def initialize(self):
        llm_hidden_size = self.llm_config.to_dict()[self.llm_config.attribute_map['hidden_size']]
        self.action_heads_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, self._action_space_size)
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        # Get encoder's representation
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][0][0, len(tokenized_context["input_ids"])-1, :]
        else:
            model_head = forward_outputs['encoder_last_hidden_state'][0, len(tokenized_context["input_ids"]) - 1, :]

        actions_score = self.action_heads_op(model_head)
        return actions_score.cpu()


class Updater(BaseUpdater):
    def generate_prompt(self, subgoals):
        head_prompt = "Possible action of the agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        templeted_prompts = [
            ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: ',
            ' \n Goal of the agent: go to the purple box \n Observation 0: A wall 4 step forward, A red key 2 steps left and 1 step forward, A grey box 2 steps left, A purple box 1 step left and 1 step forward, A green key 2 steps right and 2 steps forward, A grey key 3 steps right and 2 steps forward, \n Action 0: ',
            ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A purple key 1 step left and 1 step forward, A yellow key 1 step left, A green ball 2 steps forward, A grey ball 1 step right and 4 steps forward, A green key 1 step right and 1 step forward, A grey ball 1 step right, A green key 2 steps right and 3 steps forward, A red box 2 steps right and 1 step forward, \n Action 1: ',
            ' \n Goal of the agent: go to the purple box \n Observation 0: A wall 4 step forward, A red key 2 steps left and 1 step forward, A grey box 2 steps left, A purple box 1 step left and 1 step forward, A green key 2 steps right and 2 steps forward, A grey key 3 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A wall 3 step forward, A red key 2 steps left, A purple box 1 step left, A green key 2 steps right and 1 step forward, A grey key 3 steps right and 1 step forward, \n Action 1: ',
            ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A purple key 1 step left and 1 step forward, A yellow key 1 step left, A green ball 2 steps forward, A grey ball 1 step right and 4 steps forward, A green key 1 step right and 1 step forward, A grey ball 1 step right, A green key 2 steps right and 3 steps forward, A red box 2 steps right and 1 step forward, \n Action 1: turn right \n Observation 2: A wall 2 step right, A green key 3 steps left and 2 steps forward, A green ball 2 steps left, A red box 1 step left and 2 steps forward, A green key 1 step left and 1 step forward, A grey ball 1 step forward, \n Action 2: ',
            ' \n Goal of the agent: go to the purple box \n Observation 0: A wall 4 step forward, A red key 2 steps left and 1 step forward, A grey box 2 steps left, A purple box 1 step left and 1 step forward, A green key 2 steps right and 2 steps forward, A grey key 3 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A wall 3 step forward, A red key 2 steps left, A purple box 1 step left, A green key 2 steps right and 1 step forward, A grey key 3 steps right and 1 step forward, \n Action 1: go forward \n Observation 2: A wall 2 step forward, A wall 3 step left, A green key 2 steps right, A grey key 3 steps right, \n Action 2: ']
        for j in range(6):
            templeted_prompts[j] = head_prompt + templeted_prompts[j]
        return templeted_prompts

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        # If asked, only do embedding weights loading
        if "load_embedding" in kwargs and kwargs["load_embedding"] and not hasattr(self, "is_embedding_loaded"):
            pretrained_weights = torch.load(kwargs["llm_path"] + "/pytorch_model.bin")
            state_dict = OrderedDict({
                k: v for k, v in pretrained_weights.items() if "embed" in k or "shared" in k # Warning: this may fail if the model shares other things than embedding weights
            })
            self._llm_module.module._LLM_model.load_state_dict(state_dict, strict=False)
            self.is_embedding_loaded = True
            return {}

        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self._llm_module.parameters(), kwargs["lr"],
                                              (kwargs["beta1"], kwargs["beta2"]),
                                              eps=kwargs["adam_eps"])

        sb = {}
        for k in ['action', 'value', 'log_prob', 'advantage', 'returnn']:
            sb[k] = kwargs["exps"][k][_current_batch_ids]

        # Compute loss
        output = self._llm_module([kwargs["scoring_module_key"], 'value'],
                                  contexts=contexts, candidates=candidates, require_grad=True)
        scores = torch.stack([_o[kwargs["scoring_module_key"]] for _o in output])
        scores_max = torch.max(scores, dim=1)[0]
        values = torch.stack([_o["value"][0] for _o in output])

        proba_dist = []
        for j in range(len(scores)):
            if kwargs["scoring_module_key"] == "__score":
                # rescaled scores to avoid the flattening effect of softmax
                # softmax([1e-9, 1e-100, 1e-9])~[0.33, 0.33, 0.33]
                # softmax([1e-9, 1e-100, 1e-9]*1e9)~[0.4223, 0.1554, 0.4223]
                if scores_max[j] < 1e-45:
                    proba_dist.append(F.softmax(torch.ones_like(scores[j]), dim=-1).unsqueeze(dim=0))
                else:
                    proba_dist.append(F.softmax(scores[j] / scores_max[j], dim=-1).unsqueeze(dim=0))
            else:
                proba_dist.append(F.softmax(scores[j], dim=-1).unsqueeze(dim=0))

        proba_dist = torch.cat(proba_dist)
        dist = Categorical(probs=proba_dist)

        entropy = dist.entropy().mean()
        log_prob = dist.log_prob(sb['action'])
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
        ratio = torch.exp(log_prob - sb['log_prob'])
        surr1 = ratio * sb['advantage']
        surr2 = torch.clamp(ratio, 1.0 - kwargs["clip_eps"], 1.0 + kwargs["clip_eps"]) * sb['advantage']
        policy_loss = -torch.min(surr1, surr2).mean()

        value_clipped = sb['value'] + torch.clamp(values - sb['value'], -kwargs["clip_eps"], kwargs["clip_eps"])
        surr_v1 = (values - sb['returnn']).pow(2)
        surr_v2 = (value_clipped - sb['returnn']).pow(2)
        value_loss = torch.max(surr_v1, surr_v2).mean()

        loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss

        # Update actor-critic
        self.optimizer.zero_grad()
        """print(policy_loss.detach().item())
        print(value_loss.detach().item())
        print(" ")"""
        loss.backward()
        grad_norm = sum(
            p.grad.data.detach().cpu().norm(2) ** 2 for p in self._llm_module.parameters() if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(self._llm_module.parameters(), kwargs["max_grad_norm"])
        self.optimizer.step()

        dict_return = {"loss": loss.item(),
                       "entropy": entropy.item(),
                       "policy_loss": policy_loss.item(),
                       "value_loss": value_loss.item(),
                       "grad_norm": grad_norm.item()}

        # save the model every 100 updates
        if accelerator.process_index == 1 and kwargs["lm_server_update_first_call"]:
            if kwargs["number_updates"] % 100 == 0:
                torch.save(self._llm_module.state_dict(), kwargs["saving_path_model"] +
                           "/" + kwargs["id_expe"] + "/model.checkpoint")
                torch.save(self.optimizer.state_dict(), kwargs["saving_path_model"] +
                           "/" + kwargs["id_expe"] + "/optimizer.checkpoint")

        # save the proba_dist over actions every 50 updates
        if accelerator.process_index == 1 and kwargs["lm_server_update_first_call"]:
            if kwargs["number_updates"] % 50 == 0 and candidates is not None:
                prompts = self.generate_prompt(candidates[0])
                subgoals = [candidates[0] for i in range(6)]

                # Avoid calling DDP model and get stuck gathering buffers from all LLMs
                output = self._llm_module.module([kwargs["scoring_module_key"], 'value'],
                                                 contexts=prompts, candidates=subgoals, require_grad=False)
                scores = torch.stack([_o[kwargs["scoring_module_key"]] for _o in output])
                scores_max = torch.max(scores, dim=1)[0]

                proba_dist = []
                for j in range(len(scores)):
                    if scores_max[j] < 1e-45:
                        proba_dist.append(F.softmax(torch.ones_like(scores[j]), dim=-1).unsqueeze(dim=0))
                    else:
                        proba_dist.append(F.softmax(scores[j] / scores_max[j], dim=-1).unsqueeze(dim=0))

                proba_dist = list(torch.cat(proba_dist).cpu().numpy().flatten())

                csv_distrib_path = os.path.join(kwargs["experiment_path"], 'distrib.csv')
                csv_writer = csv.writer(open(csv_distrib_path, 'a', 1))
                csv_writer.writerow(proba_dist)

        return dict_return


def run_agent(args, algo, id_expe):
    header = (["update", "episodes", "frames", "FPS", "duration"]
              + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["success_rate"]
              + ["reshaped_return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["reshaped_return_bonus_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["entropy", "policy_loss", "value_loss", "loss", "grad_norm"])

    experiment_path = os.path.join(args.saving_path_logs, id_expe)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    csv_path = os.path.join(experiment_path, 'log.csv')
    # we don't buffer data going in the csv log, because we assume
    # that one update will take much longer than one write to the log
    first_created = not os.path.exists(csv_path)
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Restore training status
    status_path = os.path.join(experiment_path, 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                  'num_episodes': 0,
                  'num_frames': 0}

    format_str = ("\nUpdate: {} | Episodes Done: {} | Frames Seen: {:06} | FPS: {:04.0f} | Ellapsed: {}\
                               \nReward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Success Rate: {: .2f}\
                               \nReshaped: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Bonus: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f})\
                               \nFrames/Eps: {:.1f} +- {:.1f}  (Min: {}, Max {})\
                               \nEntropy: {: .3f} | Policy Loss: {: .3f} | Value Loss: {: .5f} | Loss: {: .3f} | Grad Norm: {: .3f}")

    total_start_time = time.time()
    while True:
        update_start_time = time.time()
        algo.number_updates = status['i']
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        reshaped_return_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        reshaped_return_bonus_per_episode = utils.synthesize(logs["reshaped_return_bonus_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *reshaped_return_per_episode.values(),
                *reshaped_return_bonus_per_episode.values(),
                *num_frames_per_episode.values(),
                logs["entropy"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]

        logger.info(Fore.YELLOW + format_str.format(*data) + Fore.RESET)
        csv_writer.writerow(data)

        with open(status_path, 'w') as dst:
            json.dump(status, dst)


# This will be overriden by lamorel's launcher if used
@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # lm server
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        custom_lamorel_module_functions = {
            'value': ValueModuleFn(config_args.lamorel_args.llm_args.model_type)
        }
        if config_args.rl_script_args.use_action_heads:
            custom_lamorel_module_functions['policy_head'] = ActionHeadsModuleFn(
                config_args.lamorel_args.llm_args.model_type,
                config_args.rl_script_args.size_action_space
            )
            lamorel_scoring_module_key = "policy_head"
        else:
            lamorel_scoring_module_key = "__score"

        lamorel_init()
        lm_server = Caller(config_args.lamorel_args, custom_updater_class=Updater,
                           custom_module_functions=custom_lamorel_module_functions)

    # Env
    name_env = config_args.rl_script_args.name_environment
    seed = config_args.rl_script_args.seed
    envs = []
    subgoals = []
    number_envs = config_args.rl_script_args.number_envs
    for i in range(number_envs):
        env = gym.make(name_env)
        env.seed(100 * seed + i)
        envs.append(env)
        if config_args.rl_script_args.size_action_space == 3:  # TODO update later to make it more general
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

    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        if not config_args.lamorel_args.llm_args.pretrained and config_args.rl_script_args.load_embedding:
            lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                             [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                             load_embedding=True, llm_path=config_args.lamorel_args.llm_args.model_path)

    algo = DRRN_Agent(envs, subgoals, reshape_reward, config_args.rl_script_args.spm_path, max_steps=number_envs*4)
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        algo = babyai.rl.PPOAlgoLlm(envs, lm_server, lamorel_scoring_module_key,
                                    config_args.lamorel_args.distributed_setup_args.n_llm_processes,
                                    config_args.rl_script_args.frames_per_proc,
                                    config_args.rl_script_args.discount, config_args.rl_script_args.lr,
                                    config_args.rl_script_args.beta1, config_args.rl_script_args.beta2,
                                    config_args.rl_script_args.gae_lambda, config_args.rl_script_args.entropy_coef,
                                    config_args.rl_script_args.value_loss_coef, config_args.rl_script_args.max_grad_norm,
                                    config_args.rl_script_args.adam_eps, config_args.rl_script_args.clip_eps,
                                    config_args.rl_script_args.epochs, config_args.rl_script_args.batch_size,
                                    reshape_reward,
                                    config_args.rl_script_args.name_experiment,
                                    config_args.rl_script_args.saving_path_model,
                                    config_args.rl_script_args.saving_path_logs, number_envs, subgoals, id_expe)
    run_agent(config_args.rl_script_args, algo, id_expe)
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        lm_server.close()


if __name__ == '__main__':
    main()
