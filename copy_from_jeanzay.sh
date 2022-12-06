#!/bin/bash

# SOURCECODE=uez56by@jeanzay:/gpfswork/rech/imi/uez56by/code/DLP/
SOURCECODE=uez56by@jeanzay:/gpfswork/rech/imi/ucy39hi/DLP/storage/logs/llm_gtl_nbr_env_32_Flan_T5small_action-heads_trained-embedding_nbr_actions_3_shape_reward_beta_0_seed_*

# DESTINATIONCODE=/home/tcarta/DLP
DESTINATIONCODE=/home/tcarta/DLP/storage/

echo Transfer code ...
rsync -azvh -e "ssh -i $HOME/.ssh/id_rsa" --exclude-from=exclude_rsync_from_jeanzay.txt $SOURCECODE $DESTINATIONCODE


