import numpy as np
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import DRRN
from .utils.memory import PrioritizedReplayMemory, Transition, State
import sentencepiece as spm

import babyai.rl
from dlp.test_llm import BaseAlgo

# Accelerate
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.state.device

class DRRN_Agent:
    def __init__(self, envs, subgoals, reshape_reward, spm_path, gamma=0.9, batch_size=64, memory_size=5000000,
                 priority_fraction=0, clip=5, embedding_dim=128, hidden_dim=128, lr=0.0001, max_steps=64,
                 number_epsiodes_test=0):
        super().__init__()
        self.envs = envs
        self.subgoals = subgoals
        self.reshape_reward = reshape_reward
        self.gamma = gamma
        self.batch_size = batch_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        ## self.memory = ReplayMemory(memory_size)     ## PJ: Changing to more memory efficient memory, since the pickle files are enormous
        self.memory = PrioritizedReplayMemory(capacity=memory_size,
                                              priority_fraction=priority_fraction)  ## PJ: Changing to more memory efficient memory, since the pickle files are enormous
        self.clip = clip
        self.network = DRRN(len(self.sp), embedding_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.max_steps = max_steps
        
        # Stateful env
        obs, infos = self.envs.reset()
        self.obs = obs
        self.n_envs = len(obs)
        self.obs_queue = [deque([], maxlen=3) for _ in range(self.n_envs)]
        self.acts_queue = [deque([], maxlen=2) for _ in range(self.n_envs)]
        for j in range(self.n_envs):
            self.obs_queue[j].append(infos[j]['descriptions'])
        prompts = [babyai.rl.PPOAlgoLlm.generate_prompt(goal=obs[j]['mission'], subgoals=self.subgoals[j],
                                                        deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                   for j in range(self.n_envs)]
        self.states = self.build_state(prompts)
        self.encoded_actions = self.encode_actions(self.subgoals)
        self.logs = {
            "return_per_episode": [],
            "reshaped_return_per_episode": [],
            "reshaped_return_bonus_per_episode": [],
            "num_frames_per_episode": [],
            "num_frames": self.max_steps,
            "episodes_done": 0,
            "entropy": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "grad_norm": 0,
            "loss": 0
        }
        self.returns = [0 for _ in range(self.n_envs)]
        self.reshaped_returns = [0 for _ in range(self.n_envs)]
        self.frames_per_episode = [0 for _ in range(self.n_envs)]

        self.number_episodes = number_epsiodes_test

    def observe(self, state, act, rew, next_state, next_acts, done):
        # self.memory.push(state, act, rew, next_state, next_acts, done)     # When using ReplayMemory
        self.memory.push(False, state, act, rew, next_state, next_acts,
                         done)  # When using PrioritizedReplayMemory (? PJ)

    def build_state(self, obs):
        return [State(self.sp.EncodeAsIds(o)) for o in obs]

    def encode_actions(self, acts):
        return [self.sp.EncodeAsIds(a) for a in acts]

    def act(self, states, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        act_values = self.network.forward(states, poss_acts)
        if sample:
            act_probs = [F.softmax(vals, dim=0) for vals in act_values]
            act_idxs = [torch.multinomial(probs, num_samples=1).item() \
                        for probs in act_probs]
        else:
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]

        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(act_idxs)]
        return act_ids, act_idxs, act_values


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1 - torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute Huber loss
        loss = F.smooth_l1_loss(qvals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return loss

    def update_parameters(self):
        episodes_done = 0
        for i in tqdm(range(self.max_steps // self.n_envs), ascii=" " * 9 + ">", ncols=100):
            action_ids, action_idxs, _ = self.act(self.states, self.encoded_actions, sample=True)
            actions = [_subgoals[idx] for _subgoals, idx in zip(self.subgoals, action_idxs)]
            if len(self.subgoals[0]) > 6:
                # only useful when we test the impact of the number of actions
                real_a = np.copy(action_idxs)
                real_a[real_a > 6] = 6
                obs, rewards, dones, infos = self.envs.step(real_a)
            else:
                obs, rewards, dones, infos = self.envs.step(action_idxs)
            reshaped_rewards = [self.reshape_reward(reward=r)[0] for r in rewards]
            for j in range(self.n_envs):
                self.returns[j] += rewards[j]
                self.reshaped_returns[j] += reshaped_rewards[j]
                self.frames_per_episode[j] += 1
                if dones[j]:
                    episodes_done += 1
                    self.logs["num_frames_per_episode"].append(self.frames_per_episode[j])
                    self.frames_per_episode[j] = 0
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
                    self.logs["reshaped_return_per_episode"].append(self.reshaped_returns[j])
                    self.logs["reshaped_return_bonus_per_episode"].append(self.reshaped_returns[j])
                    self.reshaped_returns[j] = 0
                    # reinitialise memory of past observations and actions
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                else:
                    self.acts_queue[j].append(actions[j])
                    self.obs_queue[j].append(infos[j]['descriptions'])

            next_prompts = [babyai.rl.PPOAlgoLlm.generate_prompt(goal=obs[j]['mission'], subgoals=self.subgoals[j],
                                                                 deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                            for j in range(self.n_envs)]
            next_states = self.build_state(next_prompts)
            for state, act, rew, next_state, next_poss_acts, done in \
                    zip(self.states, action_ids, reshaped_rewards, next_states, self.encoded_actions, dones):
                self.observe(state, act, rew, next_state, next_poss_acts, done)
            self.states = next_states
            # self.logs["num_frames"] += self.n_envs

        loss = self.update()
        if loss is not None:
            self.logs["loss"] = loss.detach().cpu().item()

        logs = {}
        for k, v in self.logs.items():
            if isinstance(v, list):
                logs[k] = v[:-episodes_done]
            else:
                logs[k] = v
        logs["episodes_done"] = episodes_done
        return logs

        # # Collect experiences
        # exps, logs = self.collect_experiences(debug=self.debug)
        # lm_server_update_first_call = True
        # for _ in tqdm(range(self.epochs), ascii=" " * 9 + "<", ncols=100):
        #     # Initialize log values
        #
        #     log_entropies = []
        #     log_policy_losses = []
        #     log_value_losses = []
        #     log_grad_norms = []
        #
        #     log_losses = []
        #
        #     # Create minibatch of size self.batch_size*self.nbr_llms
        #     # each llm receive a batch of size batch_size
        #     for inds in self._get_batches_starting_indexes():
        #         # inds is a numpy array of indices that correspond to the beginning of a sub-batch
        #         # there are as many inds as there are batches
        #
        #         exps_batch = exps[inds]
        #
        #         # return the list of dict_return calculate by each llm
        #         list_dict_return = self.lm_server.update(exps_batch.prompt,
        #                                                  self.filter_candidates_fn(exps_batch.subgoal),
        #                                                  exps=dict(exps_batch),
        #                                                  lr=self.lr,
        #                                                  beta1=self.beta1,
        #                                                  beta2=self.beta2,
        #                                                  adam_eps=self.adam_eps,
        #                                                  clip_eps=self.clip_eps,
        #                                                  entropy_coef=self.entropy_coef,
        #                                                  value_loss_coef=self.value_loss_coef,
        #                                                  max_grad_norm=self.max_grad_norm,
        #                                                  nbr_llms=self.nbr_llms,
        #                                                  id_expe=self.id_expe,
        #                                                  lm_server_update_first_call=lm_server_update_first_call,
        #                                                  saving_path_model=self.saving_path_model,
        #                                                  experiment_path=self.experiment_path,
        #                                                  number_updates=self.number_updates,
        #                                                  scoring_module_key=self.llm_scoring_module_key)
        #
        #         lm_server_update_first_call = False
        #
        #         log_losses.append(np.mean([d["loss"] for d in list_dict_return]))
        #         log_entropies.append(np.mean([d["entropy"] for d in list_dict_return]))
        #         log_policy_losses.append(np.mean([d["policy_loss"] for d in list_dict_return]))
        #         log_value_losses.append(np.mean([d["value_loss"] for d in list_dict_return]))
        #         log_grad_norms.append(np.mean([d["grad_norm"] for d in list_dict_return]))
        #
        # # Log some values
        #
        # logs["entropy"] = np.mean(log_entropies)
        # logs["policy_loss"] = np.mean(log_policy_losses)
        # logs["value_loss"] = np.mean(log_value_losses)
        # logs["grad_norm"] = np.mean(log_grad_norms)
        # logs["loss"] = np.mean(log_losses)
        #
        # return logs

    # def _get_batches_starting_indexes(self):
    #     """Gives, for each batch, the indexes of the observations given to
    #     the model and the experiences used to compute the loss at first.
    #     Returns
    #     -------
    #     batches_starting_indexes : list of lists of int
    #         the indexes of the experiences to be used at first for each batch
    #     """
    #
    #     indexes = np.arange(0, self.num_frames)
    #     indexes = np.random.permutation(indexes)
    #
    #     num_indexes = self.batch_size
    #     batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]
    #
    #     return batches_starting_indexes

    def generate_trajectories(self, dict_modifier, language='english'):

        if language == "english":
            generate_prompt = BaseAlgo.generate_prompt_english
            subgoals = self.subgoals
        elif language == "french":
            dico_traduc_act = {'turn_left': "tourner à gauche",
                               "turn_right": "tourner à droite",
                               "go_forward": "aller tout droit",
                               "eat": "manger",
                               "dance": "dancer",
                               "sleep": "dormir",
                               "do_nothing": "ne rien faire",
                               "cut": "couper",
                               "think": "penser"}
            generate_prompt = BaseAlgo.generate_prompt_french
            subgoals = [[BaseAlgo.prompt_modifier(sg, dico_traduc_act) for sg in sgs] for sgs in self.subgoals]

        episodes_done = 0
        pbar = tqdm(range(self.number_episodes), ascii=" " * 9 + ">", ncols=100)
        while episodes_done < self.number_episodes:
            # Do one agent-environment interaction

            prompts = [BaseAlgo.prompt_modifier(generate_prompt(goal=self.obs[j]['mission'], subgoals=self.subgoals[j],
                                                           deque_obs=self.obs_queue[j],
                                                           deque_actions=self.acts_queue[j]), dict_modifier)
                      for j in range(self.n_envs)]
            self.states = self.build_state(prompts)
            action_ids, action_idxs, _ = self.act(self.states, self.encoded_actions, sample=True)
            actions = [_subgoals[idx] for _subgoals, idx in zip(self.subgoals, action_idxs)]

            if len(self.subgoals[0]) > 6:
                # only useful when we test the impact of the number of actions
                real_a = np.copy(action_idxs)
                real_a[real_a > 6] = 6
                obs, rewards, dones, infos = self.envs.step(real_a)
            else:
                obs, rewards, dones, infos = self.envs.step(action_idxs)
            reshaped_rewards = [self.reshape_reward(reward=r)[0] for r in rewards]

            for j in range(self.n_envs):
                self.returns[j] += rewards[j]
                self.reshaped_returns[j] += reshaped_rewards[j]
                self.frames_per_episode[j] += 1
                if dones[j]:
                    episodes_done += 1
                    pbar.update(1)
                    self.logs["num_frames_per_episode"].append(self.frames_per_episode[j])
                    self.frames_per_episode[j] = 0
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
                    self.logs["reshaped_return_per_episode"].append(self.reshaped_returns[j])
                    self.logs["reshaped_return_bonus_per_episode"].append(self.reshaped_returns[j])
                    self.reshaped_returns[j] = 0
                    # reinitialise memory of past observations and actions
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                else:
                    self.acts_queue[j].append(actions[j])
                    self.obs_queue[j].append(infos[j]['descriptions'])

            self.obs = obs
            """next_prompts = [babyai.rl.PPOAlgoLlm.generate_prompt(goal=obs[j]['mission'], subgoals=self.subgoals[j],
                                                                 deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                            for j in range(self.n_envs)]
            next_states = self.build_state(next_prompts)

            self.states = next_states"""
            # self.logs["num_frames"] += self.n_envs
        pbar.close()

        logs = {}
        for k, v in self.logs.items():
            if isinstance(v, list):
                logs[k] = v[:-episodes_done]
            else:
                logs[k] = v
        logs["episodes_done"] = episodes_done
        return logs