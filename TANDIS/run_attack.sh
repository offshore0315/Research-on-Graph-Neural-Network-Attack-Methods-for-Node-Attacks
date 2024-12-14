#!/bin/bash

data_folder=data/split

dataset=cora
base_mod=Deep-sage
down_task=node_classification
budget=1

transf_mod=Deep-gin
transf_task=unsupervised
preT_mod=Deep-gin
saved_model=attack_models/${down_task}/$dataset/model-${base_mod}
saved_emb_model=attack_models/${transf_task}/$dataset/model-${transf_mod}

directed=0
lr=0.01
batch_size=5
reward_type=emb_silh
reward_state=marginal
# gm=mean_field
num_hops=2 
mu_type=e2e_embeds
# location for pre-trained embeddings
embeds=attack_models/emb_models/$dataset/model-${preT_mod}.npy
dqn_hidden=16
embed_dim=16
discount=0.9
q_nstep=2
num_epds=10
sample_size=5

# output_base=results/target_nodes/${dataset}-${directed}-${transf_mod}-${transf_task}-ad-more
output_base=results/target_nodes/${dataset}-${directed}-${transf_mod}-${transf_task}

if [[ "$mu_type" == "preT_embeds" ]]; then 
	save_fold=rl-${lr}-${discount}_mu-preT-${preT_mod}_q-${q_nstep}-${dqn_hidden}
else
	save_fold=rl-${lr}-${discount}_mu-${mu_type}_q-${q_nstep}-${dqn_hidden}
fi
output_root=$output_base/$save_fold

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

#directed=0
#export CUDA_VISIBLE_DEVICES=1
python main.py \
    -directed $directed \
    -budget $budget \
    -data_folder $data_folder \
    -dataset $dataset \
    -down_task $down_task \
    -saved_model $saved_model \
    -saved_emb_model $saved_emb_model \
    -save_dir $output_root \
    -embeds $embeds \
    -learning_rate $lr \
    -num_hops $num_hops \
    -mu_type $mu_type \
    -embed_dim $embed_dim \
    -dqn_hidden $dqn_hidden \
    -reward_type $reward_type \
    -reward_state $reward_state \
    -batch_size $batch_size \
    -num_epds $num_epds \
    -sample_size $sample_size \
    -q_nstep $q_nstep \
    -discount $discount \
    -phase test \
    -seed 123 \
    -nprocs 7 \
    -lcc \
    -save_sols_only \
    # -save_sols_file sols_gcn \
    # -target_perc 0.1 \
    # -device cuda \
    $@
