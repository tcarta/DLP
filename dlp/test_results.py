import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_test_results():
    root = '/home/tcarta/DLP/storage/logs'
    list_dir = os.listdir(root)

    for test_name in ['no_modification_test', 'other_name_same_categories', 'adj_synonym', 'no_meaning_nouns',
                      'no_meaning_adj', 'no_meaning_words', 'change_intro_first_personne_speaker',
                      'change_intro_first_personne_agent']:

        print('NAME TESTS: {}'.format(test_name))
        l = []
        for model_name in ['.*llm_mtrl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0.*']:
            for directory in list_dir:
                if re.match(model_name, directory):
                    # a = np.load(root+'/'+directory+'/test'+'/return_per_episode/'+test_name+'.npy')
                    # print(a)
                    l.append(np.load(root+'/'+directory+'/test'+'/BabyAI-PickUpSeqGoToLocal-v0'+'/return_per_episode/'+test_name+'.npy'))
        reward_array = np.concatenate(l)
        sr_array = (reward_array > 0).astype(int)
        """plt.hist(reward_array, bins=100)
        plt.title(test_name)
        plt.show()"""

        print("For {} the mean return per episode is {} +- {}".format(test_name, np.mean(reward_array), np.std(reward_array)))
        print("For {} the mean success rate per episode is {} +- {}".format(test_name, np.mean(sr_array), np.std(sr_array)))

print_test_results()