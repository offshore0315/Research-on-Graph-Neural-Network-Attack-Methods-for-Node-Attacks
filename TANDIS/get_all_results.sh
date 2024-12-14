#!/bin/bash

budget=20
nprocs=10
seed=123

declare -a baselines=("degrees")
declare -a datasets=("cora" "citeseer" "pubmed")
# declare -a saved_sols_list=("sols_lcc_20") #_1" "sols_lcc_20_2" "sols_lcc_20_3" "sols_lcc_20_4" "sols_lcc_20_5")
declare -a down_tasks=("link_pair" "link_prediction") #("link_pair") #("node_classification") #("link_prediction") # #("node_classification") #("link_prediction" "link_pair")
declare -a basemods=("Deep-gat") #("Deep-gcn" "Deep-sage") #("PGNN-featpre") #("Deep-gat" "GNNGuard-gcn") #("PGNN") #("Deep-gat" "GNNGuard-gcn") #("Deep-gcn" "Deep-sage") #("Deep-gat" "GNNGuard-gcn") # "Deep-gat" "GNNGuard-gcn" "PGNN-featpre") #("GNNGuard-gcn", "PGNN") #("Deep-gcn" "Deep-sage" "Deep-gat" "GNNGuard-gcn" "PGNN-featpre")

for baseline in "${baselines[@]}"; do
    for dataset in "${datasets[@]}"; do
        for down_task in "${down_tasks[@]}"; do
            for basemod in "${basemods[@]}"; do
                saved_model=attack_models/${down_task}/${dataset}/model-${basemod}
                python eval_attack.py -method ${baseline} \
                    -base_model ${saved_model} \
                    -dataset ${dataset} \
                    -down_task ${down_task} \
                    -budget ${budget} \
                    -nprocs ${nprocs} \
                    -seed ${seed} \
                    -lcc \
                    -sols_type txt \
                >> baseline_results/${baseline}/${down_task}/${dataset}/out_${basemod}_lcc.txt
            done
        done
    done
done