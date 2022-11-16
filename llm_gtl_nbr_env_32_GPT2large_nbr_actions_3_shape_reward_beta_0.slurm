#!/bin/bash
#SBATCH --job-name=llm_gtl_nbr_env_32_GPT2large_nbr_actions_3_shape_reward_beta_0_seed_%a 				# job name
#SBATCH --time=00:20:00             										        # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/llm_gtl_nbr_env_32_GPT2large_nbr_actions_3_shape_reward_beta_0_seed_%a-%j.out                # output file name
#SBATCH --error=slurm_logs/llm_gtl_nbr_env_32_GPT2large_nbr_actions_3_shape_reward_beta_0_seed_%a-%j.err                 # err file name
#SBATCH --account=imi@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4

#SBATCH --array=1-2

module purge
module load python/3.8.2
conda activate dlp

chmod +x dlp/slurm/launcher.sh

srun dlp/slurm/launcher.sh \
                    rl_script_args.path=$WORK/code/DLP/dlp/main.py \
                    rl_script_args.seed=${SLURM_ARRAY_TASK_ID} \
                    rl_script_args.number_envs=32 \
                    rl_script_args.size_action_space=3 \
                    rl_script_args.saving_path_logs=$WORK/code/DLP/storage/logs \
                    rl_script_args.name_experiment='llm_gtl' \
                    rl_script_args.name_model='GPT2large' \
                    rl_script_args.saving_path_model=$SCRATCH/DLP/models \
                    lamorel_args.llm_args.model_type=causal \
                    lamorel_args.llm_args.model_path=/gpfsscratch/rech/imi/ucy39hi/saycan-scienceworld/llms/gpt2-large \
                    lamorel_args.llm_args.model_parallelism_size=4 \
                    lamorel_args.llm_args.minibatch_size=3 \
                    lamorel_args.accelerate_args.num_machines=4 \
                    lamorel_args.accelerate_args.num_processes=5 \
                    --config-path=$WORK/code/DLP/dlp/configs \
                    --config-name=multi-node_slurm_cluster_config
                    
